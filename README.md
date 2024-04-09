# nba_stats_analyzer
App designed to use the current day's NBA schedule, collect/analyze various team and player stats, and send an email to a list of recipients

# NBA Stats Analyzer

App designed to use the current date's NBA schedule, collect/analyze various team and player stats, and send an email to a list of recipients.

## Installation

1. Download the "nba_stats_analyzer_python" folder.  
2. Open project in python IDE.  
3. Run the following command
```bash
pip install -r requirements.txt
```
4. Set all environment variables:
  - SENDER_EMAIL: (str) email account that automated email message will be sent from
  - EMAIL_PASSWORD: (str) password for the sender's email
  - RECIPEINT_EMAILS: (List["str"]) list of emails that the daily analysis will be sent to

## Usage

Run main.py daily after 10am EST

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)
