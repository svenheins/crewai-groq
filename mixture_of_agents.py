# Import packages
import os
from datetime import date, timedelta, datetime
import pandas as pd
import numpy as np
import statsapi
from crewai_tools import tool
from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq
from dotenv import load_dotenv

class MLBCrewManager:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Initialize LLM instances
        self.llm_llama70b = ChatGroq(model_name="llama3-70b-8192")
        self.llm_llama8b = ChatGroq(model_name="llama3-8b-8192")
        self.llm_gemma2 = ChatGroq(model_name="gemma2-9b-it")
        self.llm_mixtral = ChatGroq(model_name="mixtral-8x7b-32768")
        
        # Initialize agents
        self.initialize_agents()
        
        # Initialize tasks
        self.initialize_tasks()

    def initialize_agents(self):
        """Initialize all MLB agents"""
        self.mlb_researcher = Agent(
            llm=self.llm_llama70b,
            role="MLB Researcher",
            goal="Identify and return info for the MLB game related to the user prompt by returning the exact results of the get_game_info tool",
            backstory="An MLB researcher that identifies games for statisticians to analyze stats from",
            tools=[self.get_game_info],
            verbose=True,
            allow_delegation=False
        )

        self.mlb_statistician = Agent(
            llm=self.llm_llama70b,
            role="MLB Statistician",
            goal="Retrieve player batting and pitching stats for the game identified by the MLB Researcher",
            backstory="An MLB Statistician analyzing player boxscore stats for the relevant game",
            tools=[self.get_batting_stats, self.get_pitching_stats],
            verbose=True,
            allow_delegation=False
        )

        self.mlb_writer_llama = Agent(
            llm=self.llm_llama8b,
            role="MLB Writer",
            goal="Write a detailed game recap article using the provided game information and stats",
            backstory="An experienced and honest writer who does not make things up",
            tools=[],
            verbose=True,
            allow_delegation=False
        )

        self.mlb_writer_gemma = Agent(
            llm=self.llm_gemma2,
            role="MLB Writer",
            goal="Write a detailed game recap article using the provided game information and stats",
            backstory="An experienced and honest writer who does not make things up",
            tools=[],
            verbose=True,
            allow_delegation=False
        )

        self.mlb_writer_mixtral = Agent(
            llm=self.llm_mixtral,
            role="MLB Writer",
            goal="Write a detailed game recap article using the provided game information and stats",
            backstory="An experienced and honest writer who does not make things up",
            tools=[],
            verbose=True,
            allow_delegation=False
        )

        self.mlb_editor = Agent(
            llm=self.llm_llama70b,
            role="MLB Editor",
            goal="Edit multiple game recap articles to create the best final product.",
            backstory="An experienced editor that excels at taking the best parts of multiple texts to create the best final product",
            tools=[],
            verbose=True,
            allow_delegation=False
        )

    def initialize_tasks(self):
        """Initialize all MLB tasks"""
        self.collect_game_info = Task(
            description='''
            Identify the correct game related to the user prompt and return game info using the get_game_info tool. 
            Unless a specific date is provided in the user prompt, use {default_date} as the game date
            User prompt: {user_prompt}
            ''',
            expected_output='High-level information of the relevant MLB game',
            agent=self.mlb_researcher
        )

        self.retrieve_batting_stats = Task(
            description='Retrieve ONLY boxscore batting stats for the relevant MLB game',
            expected_output='A table of batting boxscore stats',
            agent=self.mlb_statistician,
            dependencies=[self.collect_game_info],
            context=[self.collect_game_info]
        )

        self.retrieve_pitching_stats = Task(
            description='Retrieve ONLY boxscore pitching stats for the relevant MLB game',
            expected_output='A table of pitching boxscore stats',
            agent=self.mlb_statistician,
            dependencies=[self.collect_game_info],
            context=[self.collect_game_info]
        )

        self.write_game_recap_llama = Task(
            description='''
            Write a game recap article using the provided game information and stats.
            Key instructions:
            - Include things like final score, top performers and winning/losing pitcher.
            - Use ONLY the provided data and DO NOT make up any information, such as specific innings when events occurred, that isn't explicitly from the provided input.
            - Do not print the box score
            ''',
            expected_output='An MLB game recap article',
            agent=self.mlb_writer_llama,
            dependencies=[self.collect_game_info, self.retrieve_batting_stats, self.retrieve_pitching_stats],
            context=[self.collect_game_info, self.retrieve_batting_stats, self.retrieve_pitching_stats]
        )

        self.write_game_recap_gemma = Task(
            description='''
            Write a game recap article using the provided game information and stats.
            Key instructions:
            - Include things like final score, top performers and winning/losing pitcher.
            - Use ONLY the provided data and DO NOT make up any information, such as specific innings when events occurred, that isn't explicitly from the provided input.
            - Do not print the box score
            ''',
            expected_output='An MLB game recap article',
            agent=self.mlb_writer_gemma,
            dependencies=[self.collect_game_info, self.retrieve_batting_stats, self.retrieve_pitching_stats],
            context=[self.collect_game_info, self.retrieve_batting_stats, self.retrieve_pitching_stats]
        )

        self.write_game_recap_mixtral = Task(
            description='''
            Write a succinct game recap article using the provided game information and stats.
            Key instructions:
            - Structure with the following sections:
                  - Introduction (game result, winning/losing pitchers, top performer on the winning team)
                  - Other key performers on the winning team
                  - Key performers on the losing team
                  - Conclusion (including series result)
            - Use ONLY the provided data and DO NOT make up any information, such as specific innings when events occurred, that isn't explicitly from the provided input.
            - Do not print the box score or write out the section names
            ''',
            expected_output='An MLB game recap article',
            agent=self.mlb_writer_mixtral,
            dependencies=[self.collect_game_info, self.retrieve_batting_stats, self.retrieve_pitching_stats],
            context=[self.collect_game_info, self.retrieve_batting_stats, self.retrieve_pitching_stats]
        )

        self.edit_game_recap = Task(
            description='''
            You will be provided three game recap articles from multiple writers. Take the best of
            all three to output the optimal final article.
            
            Pay close attention to the original instructions:

            Key instructions:
                - Structure with the following sections:
                  - Introduction (game result, winning/losing pitchers, top performer on the winning team)
                  - Other key performers on the winning team
                  - Key performers on the losing team
                  - Conclusion (including series result)
                - Use ONLY the provided data and DO NOT make up any information, such as specific innings when events occurred, that isn't explicitly from the provided input.
                - Do not print the box score or write out the section names

            It is especially important that no false information, such as any inning or the inning in which an event occured, 
            is present in the final product. If a piece of information is present in one article and not the others, it is probably false
            ''',
            expected_output='An MLB game recap article',
            agent=self.mlb_editor,
            dependencies=[self.collect_game_info, self.retrieve_batting_stats, self.retrieve_pitching_stats],
            context=[self.collect_game_info, self.retrieve_batting_stats, self.retrieve_pitching_stats]
        )

    @staticmethod
    @tool
    def get_game_info(game_date: str, team_name: str) -> str:
        """Gets high-level information on an MLB game.
        
        Params:
        game_date: The date of the game of interest, in the form "yyyy-mm-dd". 
        team_name: MLB team name. Both full name (e.g. "New York Yankees") or nickname ("Yankees") are valid. If multiple teams are mentioned, use the first one
        """
        sched = statsapi.schedule(start_date=game_date,end_date=game_date)
        sched_df = pd.DataFrame(sched)
        game_info_df = sched_df[sched_df['summary'].str.contains(team_name, case=False, na=False)]

        game_id = str(game_info_df.game_id.tolist()[0])
        home_team = game_info_df.home_name.tolist()[0]
        home_score = game_info_df.home_score.tolist()[0]
        away_team = game_info_df.away_name.tolist()[0]
        away_score = game_info_df.away_score.tolist()[0]
        winning_team = game_info_df.winning_team.tolist()[0]
        series_status = game_info_df.series_status.tolist()[0]

        game_info = '''
            Game ID: {game_id}
            Home Team: {home_team}
            Home Score: {home_score}
            Away Team: {away_team}
            Away Score: {away_score}
            Winning Team: {winning_team}
            Series Status: {series_status}
        '''.format(game_id = game_id, home_team = home_team, home_score = home_score, 
                   away_team = away_team, away_score = away_score, \
                    series_status = series_status, winning_team = winning_team)

        return game_info

    @staticmethod
    @tool
    def get_batting_stats(game_id: str) -> str:
        """Gets player boxscore batting stats for a particular MLB game
        
        Params:
        game_id: The 6-digit ID of the game
        """
        boxscores=statsapi.boxscore_data(game_id)
        player_info_df = pd.DataFrame(boxscores['playerInfo']).T.reset_index()

        away_batters_box = pd.DataFrame(boxscores['awayBatters']).iloc[1:]
        away_batters_box['team_name'] = boxscores['teamInfo']['away']['teamName']

        home_batters_box = pd.DataFrame(boxscores['homeBatters']).iloc[1:]
        home_batters_box['team_name'] = boxscores['teamInfo']['home']['teamName']

        batters_box_df = pd.concat([away_batters_box, home_batters_box]).merge(player_info_df, left_on = 'name', right_on = 'boxscoreName')
        return str(batters_box_df[['team_name','fullName','position','ab','r','h','hr','rbi','bb','sb']])

    @staticmethod
    @tool
    def get_pitching_stats(game_id: str) -> str:
        """Gets player boxscore pitching stats for a particular MLB game
        
        Params:
        game_id: The 6-digit ID of the game
        """
        boxscores=statsapi.boxscore_data(game_id)
        player_info_df = pd.DataFrame(boxscores['playerInfo']).T.reset_index()

        away_pitchers_box = pd.DataFrame(boxscores['awayPitchers']).iloc[1:]
        away_pitchers_box['team_name'] = boxscores['teamInfo']['away']['teamName']

        home_pitchers_box = pd.DataFrame(boxscores['homePitchers']).iloc[1:]
        home_pitchers_box['team_name'] = boxscores['teamInfo']['home']['teamName']

        pitchers_box_df = pd.concat([away_pitchers_box,home_pitchers_box]).merge(player_info_df, left_on = 'name', right_on = 'boxscoreName')
        return str(pitchers_box_df[['team_name','fullName','ip','h','r','er','bb','k','note']])

    def run_crew(self, user_prompt: str):
        """Run the MLB crew with the given user prompt"""
        default_date = datetime.now().date() - timedelta(1)  # Set default date to yesterday
        
        crew = Crew(
            agents=[
                self.mlb_researcher, self.mlb_statistician,
                self.mlb_writer_llama, self.mlb_writer_gemma,
                self.mlb_writer_mixtral, self.mlb_editor
            ],
            tasks=[
                self.collect_game_info,
                self.retrieve_batting_stats, self.retrieve_pitching_stats,
                self.write_game_recap_llama, self.write_game_recap_gemma,
                self.write_game_recap_mixtral, self.edit_game_recap
            ],
            verbose=False
        )
        
        return crew.kickoff(inputs={
            "user_prompt": user_prompt,
            "default_date": str(default_date)
        })

if __name__ == "__main__":
    # Create MLB crew manager instance
    mlb_manager = MLBCrewManager()
    
    # Run the crew with a sample prompt
    user_prompt = 'Write a recap of the Yankees game on July 14, 2024'
    result = mlb_manager.run_crew(user_prompt)
    print(result)
    