import importlib
import random
import json
from .utils.fixtures_generators import roundrobin

class ResearchController:
    """
    Wrapper class. Please call the start function after loading.
    """

    def __init__(self, configurations) -> None:
        self.game_type = "prisoners_dilemma"
        self.configurations = configurations

        # This list records all the moves made the players upto this point.
        # It will be passed alongwith payoff matrix on each turn to players.
        self.game_history = []
        self.global_history = []
        self.payoff_matrix = self.configurations[self.game_type]["payoff_matrix"]

        # Register players on the scoreboard.
        self.scoreboard = {}
        self.reset_scoreboard()

    def flush_game_history(self) -> None:
        """
        This function transfers clears the game history.

        Returns:
            None
        """
        self.game_history = []

    def reset_scoreboard(self) -> None:
        """
        This function resets the scoreboard.

        Returns:
            None
        """
        for player in self.configurations[self.game_type]["players"]:
            self.scoreboard[player] = 0

    def register_player_modules(self) -> dict:
        """
        This function loads the players and passes them back into the main program loop.

        Args:
            None

        Returns:
            dict: A dictionary containing the player modules mapped to their names.
        """
        # Initiate the modules for each player.
        player_modules = {}
        for player in self.configurations[self.game_type]["players"]:
            player_modules[player] = importlib.import_module(f".players.{player}", package=f"modules.{self.game_type}")

        return player_modules

    def generate_fixtures(self) -> dict:
        """
        This function creates the fixtures for the matches to be played between the players.

        Returns:
            dict: A dictionary which contains all the fixtures for a game.
        """
        fixtures = {}

        # Create fixtures.
        if (
            "roundrobin"
            in self.configurations[self.game_type]["fixture_settings"]["format"]
        ):
            fixtures = roundrobin(
                players=self.configurations[self.game_type]["players"]
            )
        else:
            fixtures = roundrobin(
                players=self.configurations[self.game_type]["players"]
            )

        return fixtures

    def score_moves(self, player1_move: str, player2_move: str) -> tuple[int, int]:
        """
        This function maps the results into the scoreboard.

        Args:
            player1_move (str): The move made by the first player.
            player2_move (str): The move made by the second player.

        Returns:
            tuple[int,int]: A tuple of strings containing the points to be awarded to each player.
        """
        if (player1_move == "cooperate") and (player2_move == "cooperate"):
            return (2, 2)
        elif (player1_move == "cooperate") and (player2_move == "defect"):
            return (0, 3)
        elif (player1_move == "defect") and (player2_move == "cooperate"):
            return (3, 0)
        elif (player1_move == "defect") and (player2_move == "defect"):
            return (1, 1)
        else:
            print(
                f"Invalid move by the players: Player1: {player1_move}, Player2: {player2_move}. Please fix this."
            )
            exit(0)

    def update_global_history(self, round_id: int, fixture_id: int, player1: str, player2: str) -> None:
        """
        Updates the global history with game history.

        Args:
            round_id (int): The round id for this entry.
            fixture_id (int): The fixture id for this entry.
            player1 (str): The player 1 of this fixture.
            player2 (str): The player 2 of this fixture.
        Returns:
            None
        """

        self.global_history.append(
            {
                "round_id": round_id,
                "fixture_id": fixture_id,
                "player1": player1,
                "player2": player2,
                "move_wise_data": self.game_history,
            }
        )
        return None

    def start(self) -> None:
        """
        The steps performed by this function are.
        1. Load the player modules.
        2. Generate fixtures for all games.
        3. Perform game iterations and store the result in a processable format.

        Args;
            None

        Returns:
            None
        """
        player_modules = self.register_player_modules()
        fixtures = self.generate_fixtures()
        game_iterations = random.randrange(
            start=self.configurations[self.game_type]["fixture_settings"]["min_iterations"],
            stop=self.configurations[self.game_type]["fixture_settings"]["max_iterations"],
            step=1,
        )

        for round in range(self.configurations[self.game_type]["fixture_settings"]["rounds"]):
            # Start the fixtures for this round.
            for fixture in fixtures:
                print(f"Round Id: {round}, Fixture Id: {fixture}, Players: {fixtures[fixture]}, Iterations: {game_iterations}")

                # Setup the players for this fixture.
                player1_controller = player_modules[fixtures[fixture][0]].PlayerController(
                    opponent_name=fixtures[fixture][1],
                    payoff_matrix=self.payoff_matrix,
                    game_history=self.game_history,
                    global_history=self.global_history,
                    scoreboard=self.scoreboard,
                )
                player2_controller = player_modules[fixtures[fixture][1]].PlayerController(
                    opponent_name=fixtures[fixture][0],
                    payoff_matrix=self.payoff_matrix,
                    game_history=self.game_history,
                    global_history=self.global_history,
                    scoreboard=self.scoreboard,
                )

                # Perform the game iterations for this fixture.
                for i in range(game_iterations):
                    player1_move = player1_controller.make_move()
                    player2_move = player2_controller.make_move()

                    # Award points to players based on moves.
                    player1_score, player2_score = self.score_moves(player1_move, player2_move)

                    # Update the scoreboard with new scores.
                    self.scoreboard[player1_controller.name] += player1_score
                    self.scoreboard[player2_controller.name] += player2_score

                    # Append the moves into history of this game.
                    self.game_history.append(
                        {
                            "move_number": i,
                            "player1_move": player1_move,
                            "player2_move": player2_move,
                            "player1_score": self.scoreboard[player1_controller.name],
                            "player2_score": self.scoreboard[player2_controller.name],
                        }
                    )

                # Save the game history into global history.
                self.update_global_history(
                    round_id=round,
                    fixture_id=fixture,
                    player1=player1_controller.name,
                    player2=player2_controller.name,
                )

                # Delete the player objects.
                del player1_controller
                del player2_controller
                # Delete the game history.
                self.flush_game_history()

            # Reset the scoreboard for next round.
            self.reset_scoreboard()

        # Save the history into a json file.
        print(f"Game data has been stored in the outputs folder: ./outputs/database/{self.game_type}/global_history.json")
        with open(f"./outputs/database/{self.game_type}/global_history.json", "w") as global_history_file:
            json.dump(self.global_history, global_history_file)

        return None
