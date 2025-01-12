from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict
import chess
from anthropic import Anthropic
from openai import OpenAI
from enum import Enum
from config import *
import ollama 
class ModelType(Enum):
    CHATGPT_4O = "gpt-4o"
    CLAUDE_SONNET = "Claude Sonnet 3.5"
    LLAMA_3_2 = "Llama3.2"


@dataclass
class GameMove:
    player: str
    move: str
    timestamp: float
    is_capture: bool
    is_check: bool

@dataclass
class GameStats:
    total_moves: int = 0
    capture_moves: int = 0
    check_moves: int = 0
    game_duration: float = 0
    winner: Optional[str] = None

class ChessAI(ABC):
    def __init__(self):
        self.move_history = []
        self.position_history = set()
        self.stagnation_threshold = 3  # Reduced from 4 to be more aggressive about avoiding repetition
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }

        # New scoring weights for move evaluation
        self.eval_weights = {
            'capture_value': 1.0,
            'center_control': 0.8,
            'development': 0.7,
            'king_safety': 0.9,
            'pawn_structure': 0.6,
            'piece_activity': 0.75,
            'position_progress': 1.0  # New weight for progressive moves
        }
        
        self.prompt_template = """
        You are playing chess as {color}. Current position critical analysis:
        
        MOVE HISTORY ANALYSIS:
        Previous Positions Repeated: {position_repetitions}
        Stagnation Warning: {stagnation_status}
        Position Progress Score: {position_progress}
        
        TACTICAL OPPORTUNITIES (MUST CONSIDER FIRST):
        Winning Captures Available:
        {capture_analysis}
        
        Material Status:
        {material_count}
        Material Balance: {material_balance}
        
        POSITION EVALUATION:
        Center Control: {center_control}
        Development Status: {development_status}
        King Safety: {king_safety}
        Undefended Pieces: {undefended_pieces}
        
        Legal moves by priority:
        1. WINNING CAPTURES/CHECKS (Must play if available):
        {forcing_moves}

        2. DEVELOPING MOVES (Play if no winning captures):
        {developing_moves}

        3. POSITIONAL MOVES (Last resort):
        {positional_moves}

        CRITICAL: Select ONE move from the above categories.
        Respond ONLY with the UCI notation (e.g., 'e2e4').
        
        Decision Priority:
        1. Capitalize on opponent's undefended pieces.
        2. Defend against immediate threats/mate threats.
        3. Execute winning captures/tactics.
        4. Protect your vulnerable pieces.
        5. Avoid repetitions and play to win
        6. When your pieces are captured, you must capture back.

        Best move given state of the game(UCI notation only):
        """
    
    def _get_piece_locations(self, board: chess.Board):
        """Get structured information about piece locations"""
        white_pieces = []
        black_pieces = []
        
        piece_symbols = {
            chess.PAWN: "Pawn",
            chess.KNIGHT: "Knight",
            chess.BISHOP: "Bishop",
            chess.ROOK: "Rook",
            chess.QUEEN: "Queen",
            chess.KING: "King"
        }
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                square_name = chess.square_name(square)
                piece_name = piece_symbols[piece.piece_type]
                if piece.color == chess.WHITE:
                    white_pieces.append(f"{piece_name} at {square_name}")
                else:
                    black_pieces.append(f"{piece_name} at {square_name}")
        
        return white_pieces, black_pieces

    def _get_material_count(self, board: chess.Board):
        """Calculate material count for both sides"""
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }
        white_material = 0
        black_material = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = piece_values[piece.piece_type]
                if piece.color == chess.WHITE:
                    white_material += value
                else:
                    black_material += value
        
        return f"White: {white_material} points, Black: {black_material} points"

    def _analyze_material_tension(self, board: chess.Board) -> str:
        """Analyze pieces under attack and potential captures"""
        tension_score = 0
        exchanges = []
        
        for move in board.legal_moves:
            if board.is_capture(move):
                captured_piece = board.piece_at(move.to_square)
                capturing_piece = board.piece_at(move.from_square)
                if captured_piece and capturing_piece:
                    value_diff = self._analyze_capture_value(board, move)
                    tension_score += abs(value_diff)
                    exchanges.append(f"{chess.piece_name(capturing_piece.piece_type)} x {chess.piece_name(captured_piece.piece_type)}")
        
        return f"Tension Score: {tension_score/100:.1f}, Possible Exchanges: {', '.join(exchanges[:3])}"

    def _annotate_moves(self, board: chess.Board):
        """Create annotated list of legal moves with piece information"""
        annotated_moves = []
        for move in board.legal_moves:
            piece = board.piece_at(move.from_square)
            piece_type = chess.piece_name(piece.piece_type).capitalize()
            from_square = chess.square_name(move.from_square)
            to_square = chess.square_name(move.to_square)
            capture = " captures" if board.is_capture(move) else " to"
            target_piece = board.piece_at(move.to_square)
            target_info = f" {chess.piece_name(target_piece.piece_type)}" if target_piece else ""
            
            promotion = f" promoting to {chess.piece_name(move.promotion)}" if move.promotion else ""
            check = " (check)" if board.gives_check(move) else ""
            
            annotation = f"{piece_type} {from_square}{capture}{target_info} {to_square}{promotion}{check} [{move.uci()}]"
            annotated_moves.append(annotation)
        
        return "\n".join(annotated_moves)

    def _analyze_position_repetition(self, board: chess.Board) -> dict:
        """Analyze position repetition and stagnation"""
        current_fen = board.fen().split(' ')[0]  # Only board position, ignore move counters
        self.position_history.add(current_fen)
        
        # Count recent position repetitions
        repetitions = sum(1 for pos in self.move_history[-8:] if pos == current_fen)
        
        # Analyze stagnation
        is_stagnating = repetitions >= self.stagnation_threshold
        
        # Analyze position progress
        if len(self.move_history) >= 4:
            recent_positions = self.move_history[-4:]
            unique_positions = len(set(recent_positions))
            progress_score = unique_positions / 4.0  # 1.0 means all positions were unique
        else:
            progress_score = 1.0
            
        return {
            "repetitions": repetitions,
            "is_stagnating": is_stagnating,
            "progress_score": progress_score
        }

    def _analyze_position_progress(self, board: chess.Board, move: chess.Move) -> float:
        """New method to evaluate if a move makes meaningful progress"""
        progress_score = 0.0
        
        # Check if the move develops a piece
        if chess.square_rank(move.from_square) in [0, 1, 6, 7]:
            if chess.square_rank(move.to_square) not in [0, 1, 6, 7]:
                progress_score += 100
        
        # Bonus for moves toward center
        center_distance_before = min(
            chess.square_distance(move.from_square, chess.E4),
            chess.square_distance(move.from_square, chess.D4),
            chess.square_distance(move.from_square, chess.E5),
            chess.square_distance(move.from_square, chess.D5)
        )
        center_distance_after = min(
            chess.square_distance(move.to_square, chess.E4),
            chess.square_distance(move.to_square, chess.D4),
            chess.square_distance(move.to_square, chess.E5),
            chess.square_distance(move.to_square, chess.D5)
        )
        if center_distance_after < center_distance_before:
            progress_score += 50
        
        # Penalize moves that have been played recently
        recent_moves = self.move_history[-6:] if len(self.move_history) >= 6 else self.move_history
        if move.uci() in recent_moves:
            progress_score -= 200
        
        return progress_score

    def _analyze_position_dynamism(self, board: chess.Board) -> str:
        """Analyze how dynamic/static the position is"""
        dynamic_factors = []
        dynamism_score = 0
        
        # Check center control
        center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
        center_control = sum(1 for sq in center_squares if board.is_attacked_by(board.turn, sq))
        dynamism_score += center_control * 2
        
        # Check piece mobility
        mobility = len(list(board.legal_moves))
        dynamism_score += mobility // 4
        
        # Check pawn structure tension
        pawn_moves = sum(1 for move in board.legal_moves if board.piece_at(move.from_square).piece_type == chess.PAWN)
        dynamism_score += pawn_moves
        
        # Add factors that contribute to dynamism
        if board.is_check():
            dynamic_factors.append("Check")
            dynamism_score += 5
        if any(board.is_capture(move) for move in board.legal_moves):
            dynamic_factors.append("Captures Available")
            dynamism_score += 3
        
        return f"Dynamism Score: {dynamism_score}, Factors: {', '.join(dynamic_factors)}"

    def _get_castling_rights(self, board: chess.Board):
        """Get readable castling rights"""
        rights = []
        if board.has_kingside_castling_rights(chess.WHITE):
            rights.append("White O-O")
        if board.has_queenside_castling_rights(chess.WHITE):
            rights.append("White O-O-O")
        if board.has_kingside_castling_rights(chess.BLACK):
            rights.append("Black O-O")
        if board.has_queenside_castling_rights(chess.BLACK):
            rights.append("Black O-O-O")
        return ", ".join(rights) if rights else "None"
    
    def _analyze_capture_value(self, board: chess.Board, move: chess.Move) -> int:
        """Calculate the value difference of a capture move"""
        piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }
        
        if not board.is_capture(move):
            return 0
            
        captured_piece = board.piece_at(move.to_square)
        capturing_piece = board.piece_at(move.from_square)
        
        if not captured_piece or not capturing_piece:  # Safety check
            return 0
            
        return piece_values[captured_piece.piece_type] - piece_values[capturing_piece.piece_type]

    def _calculate_development_score(self, board: chess.Board) -> str:
        """Calculate development score based on piece positioning"""
        score = 0
        developed_pieces = []
        
        # Bonus for developed pieces
        piece_development = {
            chess.KNIGHT: (2, ["b1", "g1"] if board.turn == chess.WHITE else ["b8", "g8"]),
            chess.BISHOP: (2, ["c1", "f1"] if board.turn == chess.WHITE else ["c8", "f8"]),
            chess.QUEEN: (1, ["d1"] if board.turn == chess.WHITE else ["d8"])
        }
        
        for piece_type, (value, initial_squares) in piece_development.items():
            for square_name in initial_squares:
                square = chess.parse_square(square_name)
                piece = board.piece_at(square)
                if not piece or piece.piece_type != piece_type:
                    score += value
                    developed_pieces.append(chess.piece_name(piece_type))
        
        # Bonus for castling
        if not board.has_kingside_castling_rights(board.turn):
            score += 3
            developed_pieces.append("Castled")
        
        # Penalty for blocked center pawns
        center_files = ['d', 'e']
        back_rank = '2' if board.turn == chess.WHITE else '7'
        for file in center_files:
            square_name = file + back_rank
            square = chess.parse_square(square_name)
            if board.piece_at(square) and board.piece_at(square).piece_type == chess.PAWN:
                score -= 1
        
        return f"Development Score: {score}, Developed: {', '.join(developed_pieces)}"

    def _analyze_captures(self, board: chess.Board) -> str:
        """Analyze all possible captures and sort by value"""
        captures = []
        for move in board.legal_moves:
            if board.is_capture(move):
                value_diff = self._analyze_capture_value(board, move)
                captured = board.piece_at(move.to_square)
                capturing = board.piece_at(move.from_square)
                if captured and capturing:
                    captures.append((
                        value_diff,
                        f"{chess.piece_name(capturing.piece_type).capitalize()} takes "
                        f"{chess.piece_name(captured.piece_type)} on {chess.square_name(move.to_square)} "
                        f"(value: {value_diff/100:+.1f}) [{move.uci()}]"
                    ))
        
        if not captures:
            return "No captures available"
            
        captures.sort(key=lambda x: x[0], reverse=True)
        return "\n".join(capture[1] for capture in captures)

    def _analyze_threats(self, board: chess.Board) -> str:
        """Analyze which pieces are under attack"""
        threats = []
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == board.turn:
                if board.is_attacked_by(not board.turn, square):
                    attackers = []
                    for attacker_square in board.attackers(not board.turn, square):
                        attacker = board.piece_at(attacker_square)
                        attackers.append(chess.piece_name(attacker.piece_type))
                    threats.append(
                        f"{chess.piece_name(piece.piece_type).capitalize()} on "
                        f"{chess.square_name(square)} threatened by {', '.join(attackers)}"
                    )
        
        return "\n".join(threats) if threats else "No pieces currently threatened"

    def _evaluate_capture(self, board: chess.Board, move: chess.Move) -> float:
        """Enhanced capture evaluation with positional considerations"""
        if not board.is_capture(move):
            return 0.0
            
        captured_piece = board.piece_at(move.to_square)
        capturing_piece = board.piece_at(move.from_square)
        
        if not captured_piece or not capturing_piece:
            return 0.0
            
        # Base trade value
        value_diff = self.piece_values[captured_piece.piece_type] - self.piece_values[capturing_piece.piece_type]
        
        # Additional positional considerations
        board.push(move)
        
        # Penalize if the capturing piece becomes vulnerable
        if board.is_attacked_by(not board.turn, move.to_square):
            defenders = len(list(board.attackers(board.turn, move.to_square)))
            attackers = len(list(board.attackers(not board.turn, move.to_square)))
            if attackers > defenders:
                value_diff -= self.piece_values[capturing_piece.piece_type] * 0.8
        
        # Bonus for captures that improve position
        if chess.square_file(move.to_square) in [3, 4] and chess.square_rank(move.to_square) in [3, 4]:
            value_diff += 50  # Bonus for capturing toward center
            
        board.pop()
        return value_diff

    def _categorize_moves(self, board: chess.Board):
        """Enhanced move categorization with stronger tactical awareness"""
        forcing_moves = []
        developing_moves = []
        positional_moves = []
        
        for move in board.legal_moves:
            move_str = move.uci()
            piece = board.piece_at(move.from_square)
            if not piece:
                continue
            
            # Calculate comprehensive move score
            capture_value = self._evaluate_capture(board, move)
            progress_score = self._analyze_position_progress(board, move)
            total_score = capture_value + progress_score
            
            # Test the move
            board.push(move)
            
            # Categorize based on enhanced criteria
            if capture_value > 0 or board.is_check():
                forcing_moves.append((
                    total_score,
                    f"{chess.piece_name(piece.piece_type)} "
                    f"{'captures' if board.is_capture(move) else 'checks'} "
                    f"(score: {total_score:+.1f}) [{move_str}]"
                ))
            elif progress_score > 0:
                developing_moves.append((
                    progress_score,
                    f"{chess.piece_name(piece.piece_type)} development "
                    f"(score: {progress_score:+.1f}) [{move_str}]"
                ))
            else:
                positional_moves.append((
                    total_score,
                    f"{chess.piece_name(piece.piece_type)} repositioning [{move_str}]"
                ))
            
            board.pop()
        
        # Sort all move categories by score
        forcing_moves.sort(key=lambda x: x[0], reverse=True)
        developing_moves.sort(key=lambda x: x[0], reverse=True)
        positional_moves.sort(key=lambda x: x[0], reverse=True)
        
        return {
            'forcing_moves': "\n".join(move[1] for move in forcing_moves) if forcing_moves else "None available",
            'developing_moves': "\n".join(move[1] for move in developing_moves) if developing_moves else "None available",
            'positional_moves': "\n".join(move[1] for move in positional_moves) if positional_moves else "None available"
        }

    def _analyze_defense(self, board: chess.Board) -> str:
        """Analyze defensive needs and immediate threats"""
        analysis = []
        
        # Check for immediate mate threats
        board.turn = not board.turn  # Temporarily switch sides to analyze opponent's moves
        for move in board.legal_moves:
            board.push(move)
            if board.is_checkmate():
                analysis.append(f"CRITICAL: Mate threat via {move.uci()}")
            board.pop()
        board.turn = not board.turn  # Switch back
        
        # Analyze undefended pieces
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == board.turn:
                attackers = board.attackers(not board.turn, square)
                defenders = board.attackers(board.turn, square)
                if attackers and not defenders:
                    analysis.append(
                        f"URGENT: Undefended {chess.piece_name(piece.piece_type)} on "
                        f"{chess.square_name(square)} under attack"
                    )
        
        return "\n".join(analysis) if analysis else "No immediate defensive concerns"

    def _analyze_vulnerabilities(self, board: chess.Board) -> str:
        """Analyze opponent's weaknesses"""
        vulnerabilities = []
        
        # Find undefended opponent pieces
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color != board.turn:
                attackers = board.attackers(board.turn, square)
                defenders = board.attackers(not board.turn, square)
                if not defenders and attackers:
                    vulnerabilities.append(
                        f"Undefended {chess.piece_name(piece.piece_type)} on {chess.square_name(square)}"
                    )
        
        # Find pinned pieces
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color != board.turn:
                if self._is_pinned(board, square):
                    vulnerabilities.append(
                        f"Pinned {chess.piece_name(piece.piece_type)} on {chess.square_name(square)}"
                    )
        
        return "\n".join(vulnerabilities) if vulnerabilities else "No major vulnerabilities found"

    def _analyze_king_safety(self, board: chess.Board) -> str:
        """Analyze king safety for both sides"""
        def king_zone_attacks(king_color):
            king_square = board.king(king_color)
            if king_square is None:
                return 0
            
            attack_count = 0
            for square in chess.SQUARES:
                if chess.square_distance(king_square, square) <= 2:
                    if board.is_attacked_by(not king_color, square):
                        attack_count += 1
            return attack_count

        own_king_attacks = king_zone_attacks(board.turn)
        opponent_king_attacks = king_zone_attacks(not board.turn)
        
        return (
            f"Your king safety: {own_king_attacks} attacks in king zone\n"
            f"Opponent king safety: {opponent_king_attacks} attacks in king zone"
        )

    def _is_pinned(self, board: chess.Board, square: int) -> bool:
        """Check if a piece is pinned to its king"""
        piece = board.piece_at(square)
        if not piece:
            return False
            
        color = piece.color
        king_square = board.king(color)
        if king_square is None:
            return False
            
        # Check if the piece can't move due to exposing king
        if board.is_pinned(color, square):
            return True
        return False

    def _analyze_pawn_structure(self, board: chess.Board) -> str:
        """Analyze pawn structure strengths and weaknesses"""
        analysis = []
        
        # Check for isolated pawns
        for file in range(8):
            pawns = []
            for rank in range(8):
                square = chess.square(file, rank)
                piece = board.piece_at(square)
                if piece and piece.piece_type == chess.PAWN and piece.color == board.turn:
                    pawns.append(rank)
            
            if pawns:
                # Check adjacent files for pawns
                has_neighbors = False
                for adjacent_file in [file - 1, file + 1]:
                    if 0 <= adjacent_file < 8:
                        for rank in range(8):
                            square = chess.square(adjacent_file, rank)
                            piece = board.piece_at(square)
                            if piece and piece.piece_type == chess.PAWN and piece.color == board.turn:
                                has_neighbors = True
                                break
                
                if not has_neighbors:
                    analysis.append(f"Isolated pawn on file {chess.FILE_NAMES[file]}")
        
        return "\n".join(analysis) if analysis else "Solid pawn structure"

    def _analyze_undefended_pieces(self, board: chess.Board) -> str:
        """Analyze undefended pieces for the current side"""
        undefended = []
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == board.turn:
                attackers = list(board.attackers(not board.turn, square))
                defenders = list(board.attackers(board.turn, square))
                if not defenders and attackers:
                    undefended.append(
                        f"{chess.piece_name(piece.piece_type)} on {chess.square_name(square)} "
                        f"attacked by {len(attackers)} piece(s)"
                    )
        return "\n".join(undefended) if undefended else "No undefended pieces"

    def _analyze_exposed_pieces(self, board: chess.Board) -> str:
        """Analyze exposed pieces that could become vulnerable"""
        exposed = []
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == board.turn:
                attackers = list(board.attackers(not board.turn, square))
                defenders = list(board.attackers(board.turn, square))
                if len(defenders) < len(attackers):
                    exposed.append(
                        f"{chess.piece_name(piece.piece_type)} on {chess.square_name(square)} "
                        f"({len(defenders)} defenders vs {len(attackers)} attackers)"
                    )
        return "\n".join(exposed) if exposed else "No exposed pieces"

    def _analyze_material_balance(self, board: chess.Board) -> str:
        """Analyze material balance with piece-specific details"""
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9
        }
        
        white_pieces = {piece_type: 0 for piece_type in piece_values}
        black_pieces = {piece_type: 0 for piece_type in piece_values}
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type != chess.KING:
                if piece.color == chess.WHITE:
                    white_pieces[piece.piece_type] += 1
                else:
                    black_pieces[piece.piece_type] += 1
        
        white_score = sum(count * piece_values[piece] for piece, count in white_pieces.items())
        black_score = sum(count * piece_values[piece] for piece, count in black_pieces.items())
        
        balance = white_score - black_score
        side_to_move_advantage = balance if board.turn == chess.WHITE else -balance
        
        return f"Material balance: {side_to_move_advantage:+d} ({'+' if side_to_move_advantage > 0 else ''}{side_to_move_advantage} pawns)"

    def _analyze_center_control(self, board: chess.Board) -> str:
        """Analyze control of central squares"""
        center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
        control = {chess.WHITE: 0, chess.BLACK: 0}
        
        for square in center_squares:
            white_attackers = len(list(board.attackers(chess.WHITE, square)))
            black_attackers = len(list(board.attackers(chess.BLACK, square)))
            control[chess.WHITE] += white_attackers
            control[chess.BLACK] += black_attackers
        
        side_to_move = board.turn
        opponent = not side_to_move
        return (
            f"Center control: {control[side_to_move]} squares attacked by you vs "
            f"{control[opponent]} by opponent"
        )

    def _analyze_development_status(self, board: chess.Board) -> str:
        """Analyze piece development status"""
        def count_developed_pieces(color):
            developed = 0
            back_rank = 0 if color == chess.WHITE else 7
            
            # Check knights and bishops
            for piece_type in [chess.KNIGHT, chess.BISHOP]:
                for square in chess.SQUARES:
                    piece = board.piece_at(square)
                    if (piece and piece.color == color and piece.piece_type == piece_type and
                        chess.square_rank(square) != back_rank):
                        developed += 1
            
            # Check if castled
            if board.has_castling_rights(color):
                developed += 1
                
            return developed
        
        own_developed = count_developed_pieces(board.turn)
        opponent_developed = count_developed_pieces(not board.turn)
        
        return f"Developed pieces: {own_developed} vs opponent's {opponent_developed}"

    def _create_prompt(self, fen: str) -> str:
        board = chess.Board(fen)
        moves = self._categorize_moves(board)
        position_analysis = self._analyze_position_repetition(board)
        
        return self.prompt_template.format(
            color="White" if board.turn == chess.WHITE else "Black",
            position_repetitions=position_analysis["repetitions"],
            stagnation_status="STAGNATING - Force dynamic play!" if position_analysis["is_stagnating"] else "Normal",
            position_progress=f"{position_analysis['progress_score']:.2f}",
            material_tension=self._analyze_material_tension(board),
            position_dynamism=self._analyze_position_dynamism(board),
            development_score=self._calculate_development_score(board),
            defense_analysis=self._analyze_defense(board),
            vulnerability_analysis=self._analyze_vulnerabilities(board),
            capture_analysis=self._analyze_captures(board),
            king_safety=self._analyze_king_safety(board),
            undefended_pieces=self._analyze_undefended_pieces(board),
            exposed_pieces=self._analyze_exposed_pieces(board),
            ascii_board=board,
            material_count=self._get_material_count(board),
            material_balance=self._analyze_material_balance(board),
            center_control=self._analyze_center_control(board),
            development_status=self._analyze_development_status(board),
            forcing_moves=moves['forcing_moves'],
            developing_moves=moves['developing_moves'],
            positional_moves=moves['positional_moves']
        )

    def _validate_move(self, move_str: str, board: chess.Board) -> str:
        """Enhanced move validation with board state checking"""
        try:
            # Clean up the move string
            move_str = move_str.strip().lower()
            
            # Remove common response artifacts
            prefixes = ["move:", "i choose", "my move is", "play", "'", '"', "`"]
            for prefix in prefixes:
                if move_str.startswith(prefix):
                    move_str = move_str[len(prefix):].strip()
                if move_str.endswith(prefix):
                    move_str = move_str[:-len(prefix)].strip()
            
            # Basic UCI format validation
            if not (4 <= len(move_str) <= 5):
                raise ValueError(f"Invalid move format: {move_str}")
            
            # Create chess.Move object
            try:
                move = chess.Move.from_uci(move_str)
            except ValueError:
                raise ValueError(f"Invalid UCI format: {move_str}")
            
            # Check if move is legal in current position
            if move not in board.legal_moves:
                legal_moves = [m.uci() for m in board.legal_moves]
                raise ValueError(f"Illegal move {move_str}. Legal moves are: {', '.join(legal_moves)}")
            
            return move_str
            
        except Exception as e:
            raise ValueError(f"Move validation failed: {str(e)}")

    def _is_valid_square(self, square: str) -> bool:
        """Validate if a square name is valid (e.g., 'e4', 'a1')"""
        if len(square) != 2:
            return False
        file, rank = square[0], square[1]
        return (
            file in 'abcdefgh' and
            rank in '12345678'
        )
    
    def get_move(self, fen: str) -> str:
        """Get move with position history tracking"""
        board = chess.Board(fen)
        max_retries = 3
        errors = []
        
        for attempt in range(max_retries):
            try:
                move_str = self._get_move_from_model(fen)
                validated_move = self._validate_move(move_str, board)
                
                # Track the position after making the move
                current_fen = board.fen().split(' ')[0]
                self.move_history.append(current_fen)
                
                return validated_move
            except ValueError as e:
                errors.append(f"Attempt {attempt + 1}: {str(e)}")
                continue
        
        # If we've exhausted retries, make a fallback move
        legal_moves = list(board.legal_moves)
        if legal_moves:
            fallback_move = legal_moves[0].uci()
            current_fen = board.fen().split(' ')[0]
            self.move_history.append(current_fen)
            return fallback_move
        
        raise ValueError(f"Failed to get valid move after {max_retries} attempts. Errors: {'; '.join(errors)}")

    @abstractmethod
    def _get_move_from_model(self, fen: str) -> str:
        """Implement in subclasses to get raw move from specific model"""
        pass

class OpenAIChessAI(ChessAI):
    def __init__(self, model_type: ModelType):
        super().__init__()
        self.client = OpenAI()
        self.model = model_type.value
        self.name = model_type.value

    def _get_move_from_model(self, fen: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.1,
            messages=[{
                "role": "user",
                "content": self._create_prompt(fen)
            }]
        )
        return response


class AnthropicChessAI(ChessAI):
    def __init__(self, model_type: ModelType):
        super().__init__()
        self.client = Anthropic(api_key=ANTHROPIC_API_KEY)
        self.name = model_type.value

    def _get_move_from_model(self, fen: str) -> str:
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=10,
            temperature=0.1,
            messages=[{
                "role": "user",
                "content": self._create_prompt(fen)
            }]
        )
        return response.content[0].text.strip()

class LlamaChessAI(ChessAI):
    def __init__(self, model_type: ModelType):
        super().__init__()
        self.model_name = model_type.value.lower()
        self.name = model_type.value
    
    def _get_move_from_model(self, fen: str) -> str:
        response = ollama.generate(
            model=self.model_name,
            prompt=self._create_prompt(fen)
        )
        return response['response'].strip()

