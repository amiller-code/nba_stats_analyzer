from automated_email import AutomatedEmail

import os
import json
import requests
import shutil
import pandas as pd
from datetime import datetime
from dataclasses import dataclass
from pprint import pprint
from unidecode import unidecode
from nba_api.stats.static import teams
from nba_api.stats.endpoints import playergamelogs, leaguedashteamstats
from basketball_reference_web_scraper import client

# Static dictionary to convert basketball_reference_web_scraper player positions
POSITION_DICT = {
    "[<Position.CENTER: 'CENTER'>]": "C",
    "[<Position.POWER_FORWARD: 'POWER FORWARD'>]": "PF",
    "[<Position.SHOOTING_GUARD: 'SHOOTING GUARD'>]": "SG",
    "[<Position.POINT_GUARD: 'POINT GUARD'>]": "PG",
    "[<Position.SMALL_FORWARD: 'SMALL FORWARD'>]": "SF",
}
# RECIPIENTS = [json.loads(os.environ.get("RECIPIENT_EMAILS"))[0]]    # Send to ONLY ME
RECIPIENTS = json.loads(os.environ.get("RECIPIENT_EMAILS"))         # Send to FULL LIST

pd.set_option('display.max_columns', None)
pd.set_option("display.width", 225)
pd.options.mode.chained_assignment = None
pd.options.display.float_format = '{:,.2f}'.format


@dataclass
class Team:
    abbr: str
    id: int = 0


@dataclass
class MatchUp:
    h_team: Team
    v_team: Team


@dataclass
class StatDF:
    header: str
    df: pd.DataFrame


def pull_schedule() -> pd.DataFrame:
    """Request today's NBA schedule json from cdn.nba.com and convert to dataframe."""
    schedule_r = requests.get(
        "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json"
    )
    schedule_json = schedule_r.json()["scoreboard"]["games"]
    schedule_df = pd.DataFrame(schedule_json)
    return schedule_df


def pull_player_positions() -> pd.DataFrame:
    """Use basketball_reference_web_scraper to find all current players' position on their respective team"""
    # # Basketball Reference call
    br_df = pd.DataFrame(client.players_season_totals(season_end_year=2024))
    br_df = br_df[["name", "positions", "team"]]

    # Convert positions and team columns to str since they are listed as enums
    br_df["positions"] = br_df["positions"].astype(str)
    br_df["team"] = br_df["team"].astype(str)

    # Remove accents on names to match stats.nba.com
    br_df["name"] = br_df["name"].apply(unidecode)
    # Use the static positions dictionary to convert to a more legible position string
    br_df["positions"] = br_df["positions"].apply(lambda x: POSITION_DICT.get(x))
    br_df["team"] = br_df["team"].str.split("_").apply(lambda x: x[-1]).str.lower()
    # Rename columns to match nba_api dataframe columns
    br_df.rename(columns={
        "name": "player_name",
        "team": "team_mascot",
        "positions": "position"
    }, inplace=True)

    return br_df


def pull_all_players_data() -> pd.DataFrame:
    """Use nba_api.playergamelogs to return player data using the all teams in the current season."""
    game_log = playergamelogs.PlayerGameLogs(season_nullable="2023-24",
                                             ).get_data_frames()[0]
    game_log.columns = map(str.lower, game_log.columns)
    game_log["team_mascot"] = game_log["team_name"].str.split(" ").apply(lambda x: x[-1]).str.lower()
    game_log["opponent_team_abbreviation"] = game_log["matchup"].str.split(" ").apply(lambda x: x[2])

    # time.sleep(1)
    return game_log


def pull_team_defense_data() -> pd.DataFrame:
    """Use nba_api to find the current defensive rating dataframe for each team. Columns renamed to 'opponent_...'
     since this will be used to z-scores."""
    # Request live team defensive ratings
    team_log = leaguedashteamstats.LeagueDashTeamStats(
        measure_type_detailed_defense="Defense",
        season="2023-24"
    ).get_data_frames()[0]
    team_log.columns = map(str.lower, team_log.columns)
    team_log.rename(columns={"def_rating_rank": "defense_rank",
                             "def_rating": "defense_rating"}, inplace=True)
    defense_log = team_log[["team_id", "defense_rating", "defense_rank"]]
    # Pull static teams dataframe for IDs and abbreviations
    team_details = pd.DataFrame(teams.get_teams()).rename(
        columns={"id": "team_id", "abbreviation": "team_abbreviation"})
    defense_log = pd.merge(left=defense_log, right=team_details[["team_id", "team_abbreviation"]], on="team_id")
    defense_log.drop(labels=["team_id"], axis=1, inplace=True)
    # Rename columns because this will be used for player z-score data
    defense_log.columns = ["opponent_" + x for x in defense_log.columns]

    # time.sleep(1)
    return defense_log


def find_matchup_list(schedule: pd.DataFrame) -> list[MatchUp]:
    """Use the dataframe from 'pull_schedule()' and extract the teams. Returns a list of games in the form of
     list[MatchUp(HomeTeam, VisitingTeam)]."""
    matchups_today = []
    # Search the schedule dataframe for each match-up. Columns are specific to cdn.nba.com "todaysScoreboard" json
    h_team_list = schedule["homeTeam"]
    v_team_list = schedule["awayTeam"]
    for match_i in schedule.index:
        h_team = h_team_list.iat[match_i]
        v_team = v_team_list.iat[match_i]
        matchups_today.append(
            MatchUp(
                h_team=Team(abbr=h_team["teamTricode"], id=h_team["teamId"]),
                v_team=Team(abbr=v_team["teamTricode"], id=v_team["teamId"])
            )
        )
    for match in matchups_today:
        print(match)
    return matchups_today


# Defunct method. May be needed again if cdn.nba.com cannot be used for schedule
def static_teams_to_ids(matchups_today: list[MatchUp]) -> list[MatchUp]:
    """Currently unneeded. Used for finding each team's ID when schedule is requested from stats.nbs.com."""
    teams_static = pd.DataFrame(teams.get_teams())
    for match in matchups_today:
        h_team_abbr = match.h_team.abbr
        v_team_abbr = match.v_team.abbr
        match.h_team.id = teams_static[teams_static["abbreviation"] == h_team_abbr]["id"].iloc[0]
        match.v_team.id = teams_static[teams_static["abbreviation"] == v_team_abbr]["id"].iloc[0]
    return matchups_today


def static_team_details() -> pd.DataFrame:
    pass


def simplify_stats_df(stats_df: pd.DataFrame) -> pd.DataFrame:
    """Modifies the supplied dataframe to remove unnecessary columns and calculate specific points columns.
     optional_cols variable can be modified if certain columns are needed for new charts."""
    common_cols = ["min", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta",
                   "oreb", "dreb", "reb", "ast", "tov", "stl", "blk", "blka", "pf", "pfd"]
    optional_cols = ["player_name"]
    optional_cols = [x for x in optional_cols if x in stats_df.columns]

    df = stats_df[common_cols + optional_cols]

    df["ft_pts"] = df["ftm"] * 1
    df["fg_pts"] = df["fgm"] * 2
    df["fg3_pts"] = df["fg3m"] * 3
    df["total_pts"] = df["ftm"] + df["fgm"] + df["fg3m"]

    # Only count players who have minutes or shots attempted greater than the team's median
    df["pts_per_min"] = df["total_pts"] / df["min"] * (df["min"] > df["min"].median())
    df["ft_pct"] = df["ftm"] / df["fta"] * 100 * (df["fta"] > df["fta"].median())
    df["fg_pct"] = df["fgm"] / df["fga"] * 100 * (df["fga"] > df["fga"].median())
    df["fg3_pct"] = df["fg3m"] / df["fg3a"] * 100 * (df["fg3a"] > df["fg3a"].median())

    return df


def simplify_stats_df_for_z_score(stats_df: pd.DataFrame, stats_to_norm: list[str]) -> pd.DataFrame():
    """Method for simplifying the player statistics dataframe specifically for z-score analysis"""
    # Manually split the 'matchup' column to extract the abbreviation of the opponent team (always index 2 in split list
    # because it will be formatted as either 'x vs. y' or 'x @ y'
    stats_df["opponent_team_abbreviation"] = stats_df["matchup"].str.split(" ").apply(lambda x: x[2])
    stats_df = stats_df[["player_id", "player_name", "team_id", "team_name", "team_abbreviation", "team_mascot",
                         "opponent_team_abbreviation", "game_date", "matchup"] + stats_to_norm]
    # Multiple all percentage fractions by 100
    stats_df[[col for col in stats_df.columns if "pct" in col]] = (
            stats_df[[col for col in stats_df.columns if "pct" in col]] * 100)
    return stats_df


def organize_opp_position_stats(player_df: pd.DataFrame, position_df: pd.DataFrame) -> pd.DataFrame:
    """Use a team's game log dataframe and the static positions dataframe to merge each player's position to their
     row. Return a dataframe that is grouped by player position and averaged."""
    df = pd.merge(
        left=player_df,
        right=position_df,
        on=["player_name", "team_mascot"],
        how="left"
    ).dropna(subset="position")

    # Add a row for each game_id and position that is null to account for games where any given position didn't play
    game_ids = df["game_id"].unique()
    unique_positions = df["position"].unique()
    null_positions = [[x, y] for y in unique_positions for x in game_ids]
    df = pd.concat([df, pd.DataFrame(null_positions, columns=["game_id", "position"])])

    # Find each game's statistic per position
    df_group = df.groupby(by=["game_id", "position"]).sum(numeric_only=True)
    # Find each position's average statistics
    position_stats_df = df_group.groupby(by="position").mean()
    position_stats_df = simplify_stats_df(position_stats_df)

    return position_stats_df


def organize_for_player_stats(player_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare a team's player data for finding maximum stats. Find each player's mean data and reassign the player
    names, then simplify the dataframe to remove unnecessary rows."""
    player_stats_df = player_df.groupby(by=["player_id"]).mean(numeric_only=True)
    # Re-assign player name since the numeric_only param means player_name is lost
    player_stats_df = pd.merge(left=player_stats_df, right=player_df[["player_id", "player_name"]],
                               left_index=True, right_on="player_id")
    player_stats_df = simplify_stats_df(player_stats_df)
    return player_stats_df


def organize_z_score_stats(
        players_log: pd.DataFrame, team_log: pd.DataFrame,
        z_score_list: list[str], stat_norm_list: list[str]) -> pd.DataFrame:
    """Calculate the z-scores for the opponent data (z_score_list) in the player game log dataframe.
    Use these z-scores to calculate 'normalized' player stats (stat_norm_list)."""
    z_df = pd.merge(left=team_log, right=players_log, on="opponent_team_abbreviation")

    # Find z-scores
    z_df_mean = z_df.groupby("player_id").mean(numeric_only=True)[z_score_list]
    z_df_std = z_df.groupby("player_id").std(numeric_only=True)[z_score_list]
    z_df_mean.columns = [f"{x}_mean" if x in z_score_list else x for x in z_df_mean.columns]
    z_df_std.columns = [f"{x}_std" if x in z_score_list else x for x in z_df_std.columns]
    z_df = pd.merge(left=z_df, right=z_df_mean, on="player_id")
    z_df = pd.merge(left=z_df, right=z_df_std, on="player_id")

    for col in z_score_list:
        # Calculate z-scores and invert because higher rank & rating means worse team performance
        z_df[f"z_{col}"] = -(z_df[col] - z_df[f"{col}_mean"]) / z_df[f"{col}_std"]
        z_df.drop(labels=[f"{col}_mean", f"{col}_std"], axis=1, inplace=True)
        for stat in stat_norm_list:
            # Calculate "normalized stat" based on points scored and opponent defensive z-scores
            z_df[f"normalized_{stat}-{col}"] = z_df[stat] * (1 + z_df[f"z_{col}"])
        z_df.drop(labels=[f"z_{col}"], axis=1, inplace=True)

    return z_df


def find_opp_max_stats(position_stats_df: pd.DataFrame) -> dict:
    """Use the averaged position statistics dataframe to create a dictionary with the given max stats. Reformat all
    dictionary values to a string in the form of '{Position}, {Statistic}'. Currently, finds max of: Points, Free
    throws made, Field goals made, 3-pointers made, Assists, Rebounds"""
    max_dict = {
        "most_pts": (position_stats_df["total_pts"].idxmax(),
                     position_stats_df["total_pts"][position_stats_df["total_pts"].idxmax()]),
        "most_ftm": (position_stats_df["ftm"].idxmax(),
                     position_stats_df["ftm"][position_stats_df["ftm"].idxmax()]),
        "most_fgm": (position_stats_df["fgm"].idxmax(),
                     position_stats_df["fgm"][position_stats_df["fgm"].idxmax()]),
        "most_fg3m": (position_stats_df["fg3m"].idxmax(),
                      position_stats_df["fg3m"][position_stats_df["fg3m"].idxmax()]),
        "most_ast": (position_stats_df["ast"].idxmax(),
                     position_stats_df["ast"][position_stats_df["ast"].idxmax()]),
        "most_reb": (position_stats_df["reb"].idxmax(),
                     position_stats_df["reb"][position_stats_df["reb"].idxmax()]),
    }

    max_dict = {key: f"{value[0]}; {value[1]:.2f}" for (key, value) in max_dict.items()}
    return max_dict


def find_for_player_max_stats(player_stats: pd.DataFrame) -> dict:
    """Use the averaged player statistics dataframe to create a dictionary with the given max stats. Reformat all
        dictionary values to a string in the form of '{Player Name}, {Statistic}'. Currently, finds max of: Minutes
        played, Total points, Points per minute, Field goals made, Field goals attempted, Field goal percentage,
        3-pointers made, 3-pointers attempted, 3-pointers percentage, Assists, Rebounds"""
    max_stat_list = ["min", "total_pts", "pts_per_min",
                     # "ftm", "fta", "ft_pct",
                     "fgm", "fga", "fg_pct",
                     "fg3m", "fg3a", "fg3_pct",
                     "ast", "reb"]
    max_dict = {}
    for stat in max_stat_list:
        max_i = player_stats[stat].idxmax()
        max_dict[f"most_{stat}"] = (player_stats.at[max_i, "player_name"],
                                    player_stats.at[max_i, stat])
    max_dict = {key: f"{value[0]}; {value[1]:.2f}" for (key, value) in max_dict.items()}
    return max_dict


def find_player_z_score_avg_stats(
        z_score_df: pd.DataFrame,
        z_score_list: list[str], stat_norm_list: list[str]) -> pd.DataFrame:
    """Use z-score dataframe to find each player's average normalized statistics."""
    # Create a list that will be used to slice the z_score dataframe to only the necessary columns
    stat_cols = []
    for stat in stat_norm_list:
        stat_cols.append(stat)
        stat_cols += [f"normalized_{stat}-{z_score}" for z_score in z_score_list]

    player_avg_df: pd.DataFrame = z_score_df.groupby(["player_id", "player_name"]).mean(numeric_only=True)[stat_cols]
    # Drop players who do not have z-normalized stats or have <5 points
    player_avg_df.dropna(axis=0, thresh=int(len(player_avg_df.columns) * 2 / 3), inplace=True)
    player_avg_df.drop(player_avg_df[player_avg_df["pts"] < 5].index, inplace=True)
    # Convert all values to rounded strings
    for col in player_avg_df.columns:
        player_avg_df[col] = player_avg_df[col].apply(lambda x: f"{x:.2f}")

    player_avg_df.reset_index(inplace=True)
    player_avg_df.drop(labels="player_id", axis=1, inplace=True)
    player_avg_df.set_index(keys="player_name", inplace=True)
    player_avg_df.sort_index(inplace=True)

    return player_avg_df


def analyze_opp_position_stats(match_i: int, team: Team, positions: pd.DataFrame, player_data: pd.DataFrame)\
        -> pd.DataFrame:
    """Organizer function to find which position plays best against the given team. Returns dataframe that will be
    concatenated later with each other team that plays today."""
    # Slice DataFrame to only players against the supplied team
    player_data_vs_team = player_data[player_data["opponent_team_abbreviation"] == team.abbr].copy()
    position_stats = organize_opp_position_stats(player_df=player_data_vs_team, position_df=positions)
    # team_logs.to_json(f"{team.abbr}_opp_player_logs.json")
    team_results = find_opp_max_stats(position_stats)

    team_results = {(match_i, team.abbr): team_results}
    opp_position_df = pd.DataFrame(data=team_results).transpose()
    # Cleanup column headers
    opp_position_df.columns = opp_position_df.columns.str.removeprefix("most_")

    return opp_position_df


def analyze_for_player_stats(team: Team, match_i: int, player_data: pd.DataFrame) -> pd.DataFrame:
    """Organizer function to find which player has the best stats for the given team. Returns dataframe that will be
    concatenated later with each other team that plays today."""
    # Slice DataFrame to only players for the supplied team
    player_data_for_team = player_data[player_data["team_abbreviation"] == team.abbr].copy()
    position_stats = organize_for_player_stats(player_data_for_team)
    team_results = find_for_player_max_stats(position_stats)

    team_results = {(match_i, team.abbr): team_results}
    for_team_df = pd.DataFrame(data=team_results).transpose()
    # Cleanup column headers
    for_team_df.columns = for_team_df.columns.str.removeprefix("most_")
    for_team_df.columns = for_team_df.columns.str.replace("_", " ")

    return for_team_df


def analyze_player_z_scores(team: Team, match_i: int, player_data: pd.DataFrame) -> pd.DataFrame:
    """Organizer function to find each player's 'normalized' statistics based on opponent statistics. Returns dataframe
     that will be concatenated later with each other team that plays today."""
    # Slice DataFrame to only players for the supplied team
    player_data_for_team = player_data[player_data["team_abbreviation"] == team.abbr].copy()
    # Declare which opponent stats will be used in calculating z-scores
    z_score_list = ["opponent_defense_rating", "opponent_defense_rank"]
    # Declare which player stats will be normalized
    stat_norm_list = ["pts", "fg3_pct", "fg3a", "ast", "reb"]
    # player_data.to_json(f"{team.abbr}_player_stats.json")
    player_data_for_team = simplify_stats_df_for_z_score(stats_df=player_data_for_team, stats_to_norm=stat_norm_list)

    team_defense_data = pull_team_defense_data()

    # team_defense_data.to_json(f"team_def_ratings.json")

    z_score_df = organize_z_score_stats(players_log=player_data_for_team, team_log=team_defense_data,
                                        z_score_list=z_score_list, stat_norm_list=stat_norm_list)
    z_score_df = find_player_z_score_avg_stats(z_score_df=z_score_df,
                                               z_score_list=z_score_list, stat_norm_list=stat_norm_list)

    # Set MultiIndex for final formatting
    z_score_df["Matchup"] = match_i
    z_score_df["Team"] = team.abbr
    z_score_df.reset_index(inplace=True)
    z_score_df.rename(columns={"player_name": "Player"}, inplace=True)
    z_score_df.set_index(["Matchup", "Team", "Player"], inplace=True)
    # Cleanup column headers
    z_score_df.columns = z_score_df.columns.str.replace("opponent_", "")
    z_score_df.columns = z_score_df.columns.str.replace("_", " ")
    z_score_df.columns = z_score_df.columns.str.replace("-", "; ")

    return z_score_df


def loop_matchups(matchups: list[MatchUp]) -> list[StatDF]:
    """Loop through each team in the list of today's match-ups to find the requested stats. Return a list containing a
    dataframe for each stat type. Currently, analyzes: Best position stats against each team's defense, Best player
    for each team, Each player's normalized statistic based on opponent defense z-score."""

    # Pull all necessary DataFrames
    player_df = pull_all_players_data()
    positions_df = pull_player_positions()
    # positions_df.to_json("player_positions.json")

    opp_position_final_df = pd.DataFrame()
    for_team_final_df = pd.DataFrame()
    z_score_final_df = pd.DataFrame()
    for i, match in enumerate(matchups):
        print()
        for team in match.h_team, match.v_team:
            print(team)

            # # Find the best opposing position against this team
            opp_position_df = analyze_opp_position_stats(
                match_i=i,
                team=team,
                positions=positions_df,
                player_data=player_df,
            )
            opp_position_final_df = pd.concat(objs=[opp_position_final_df, opp_position_df])
            # # Find the best stats shooter for this team
            for_team_df = analyze_for_player_stats(
                team=team,
                match_i=i,
                player_data=player_df
            )
            for_team_final_df = pd.concat(objs=[for_team_final_df, for_team_df])

            # # Find each player's z-score normalized stats
            z_score_df = analyze_player_z_scores(
                team=team,
                match_i=i,
                player_data=player_df,
            )
            z_score_final_df = pd.concat(objs=[z_score_final_df, z_score_df])

    opp_position_final_df.index.rename(names=["Matchup", "Team"], inplace=True)
    for_team_final_df.index.rename(names=["Matchup", "Team"], inplace=True)

    opp_position_stat_df = StatDF(header="Opponent Position Leaders", df=opp_position_final_df)
    for_team_stat_df = StatDF(header="Team Stat Leaders", df=for_team_final_df)
    z_score_stat_df = StatDF(header="Player Z-Normalized Stats", df=z_score_final_df)

    return [opp_position_stat_df, for_team_stat_df, z_score_stat_df]


def initiate_game_analysis(schedule_df: pd.DataFrame) -> list[str]:
    """Primary function for organizing the analysis. Function requires a dataframe with the list of games/teams for
    today. It will convert them to MatchUps, loop through them, extract the needed data, and convert the resultant
    DataFrames to HTML strings and CSVs."""
    matchups_list = find_matchup_list(schedule_df)
    # pd.DataFrame(matchups_list).to_json("Data/data_20240404/schedule.json")

    # Pull data for each team and organize
    df_list = loop_matchups(matchups=matchups_list)
    save_dfs_to_csv(df_list)
    html_list = format_dfs_to_html(df_list)
    return html_list


def prep_dfs_for_html(stat_dfs: list[StatDF]) -> list[StatDF]:
    for stat_df in stat_dfs:
        stat_df.df.replace(to_replace='; ', value='<br>', inplace=True, regex=True)

        if "Normalized" not in stat_df.header:
            continue

        # Set HTML highlighting at the beginning and end of numbers where the normalized stat is "percent_diff" larger
        # than the original stat
        percent_diff = 0.25
        for col in stat_df.df.columns:
            if not col.startswith("normalized"):
                continue
            stat_col = col.split("; ")[0].removeprefix("normalized ")
            stat_df.df["%diff"] = ((stat_df.df[col].astype(float) - stat_df.df[stat_col].astype(float)) /
                                   stat_df.df[stat_col].astype(float))
            stat_df.df["prefix"] = ""
            stat_df.df["suffix"] = ""
            stat_df.df.loc[stat_df.df["%diff"].abs() > percent_diff, "prefix"] += "<mark>"
            stat_df.df.loc[stat_df.df["%diff"].abs() > percent_diff, "suffix"] += "</mark>"
            stat_df.df[col] = stat_df.df["prefix"] + stat_df.df[col] + stat_df.df["suffix"]
            stat_df.df.drop(axis=1, columns=["%diff", "prefix", "suffix"], inplace=True)
    return stat_dfs


def format_dfs_to_html(stat_dfs: list[StatDF]):
    """Convert a list of dataframes to a list of formatted htmls."""
    html_prep_stat_dfs = prep_dfs_for_html(stat_dfs)
    df_html_list = []
    for stat_df in html_prep_stat_dfs:
        header_str = f"<h2>{stat_df.header}</h2>"
        col_len = [stat_df.df[x].str.len().max() for x in stat_df.df.columns]
        df_str = stat_df.df.to_html(justify="center", col_space=col_len)
        # Replace html escape characters to place <br> and <mark> in tables correctly
        df_str = df_str.replace("&lt;", "<")
        df_str = df_str.replace("&gt;", ">")

        df_html_list.append(header_str + df_str)
    return df_html_list


def save_dfs_to_csv(stat_dfs: list[StatDF]) -> None:
    # Check if temp folder exists and create it if not
    if not os.path.exists("../temp"):
        os.makedirs("../temp")
    # Clear out the temp folder prior to saving
    for file in os.scandir("../temp"):
        os.remove(file.path)

    for stat_df in stat_dfs:
        csv_name = stat_df.header.lower().replace(" ", "_")
        stat_df.df.to_csv(f"temp/{csv_name}.csv")


def create_email_fields(html_list: list[str]) -> None:
    """Fill in field for the global email variable and attach csv files"""
    today = datetime.today().strftime("%m-%d-%Y")

    email.message['To'] = ",".join(RECIPIENTS)
    email.message['Subject'] = f"NBA Stats {today}"
    email.html_str = "<br>".join(html_list)


if __name__ == "__main__":
    # Initialize email instance
    email = AutomatedEmail()

    # Find schedule and convert to team IDs
    games_df = pull_schedule()
    if games_df.shape[0] > 0:  # If there is at least one game today, run the analyzer
        stats_html_list = initiate_game_analysis(games_df)
        email.attach_csv()  # Only attach CSVs if there are games today
    else:  # If there are no games today, let the recipients know in the email
        stats_html_list = ["No NBA Games Today. (AWS)"]

    # Attach html list and send email
    create_email_fields(stats_html_list)
    email.attach_html()
    email.send_email()
