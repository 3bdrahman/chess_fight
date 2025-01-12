import chess
import time
from typing import Optional, List
from models import GameMove, GameStats, ChessAI

class ChessGame:
    def __init__(self, player1: ChessAI, player2: ChessAI):
        self.board = chess.Board()
        self.player1 = player1
        self.player2 = player2
        self.moves: List[GameMove] = []
        self.stats = GameStats()
        self.start_time= time.time()

    def play_move(self) -> Optional[GameMove]:
        current_player = self.player1 if len(self.moves) % 2 == 0 else self.player2
        
        try:
            move_str = current_player.get_move(self.board.fen())
            move = chess.Move.from_uci(move_str)
            
            if move in self.board.legal_moves:
                game_move = GameMove(
                    player=current_player.name,
                    move=move_str,
                    timestamp=time.time(),
                    is_capture=self.board.is_capture(move),
                    is_check=self.board.gives_check(move)
                )
                
                self.board.push(move)
                self.moves.append(game_move)
                self._update_stats(game_move)
                return game_move
                
            raise ValueError(f"Illegal move {move_str}")
            
        except Exception as e:
            raise ValueError(f"Invalid move: {str(e)}")

    def _update_stats(self, move: GameMove) -> None:
        self.stats.total_moves += 1
        if move.is_capture:
            self.stats.capture_moves += 1
        if move.is_check:
            self.stats.check_moves += 1

    @property
    def is_game_over(self) -> bool:
        if self.board.is_game_over():
            self.stats.game_duration = time.time() - self.start_time
            self.stats.winner = self._determine_winner()
            return True
        return False

    def _determine_winner(self) -> str:
        if self.board.is_checkmate():
            return self.player1.name if self.board.turn == chess.BLACK else self.player2.name
        return "Draw" if self.board.is_stalemate() or self.board.is_insufficient_material() else "Unknown"
