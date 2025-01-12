# ui.py
import streamlit as st
import chess.svg
import pandas as pd
import time
from datetime import datetime
from game import *
from models import *
class ChessUI:
    def __init__(self):
        self.board_placeholder = st.empty()
        self.stats_placeholder = st.empty()
        self.move_history_placeholder = st.empty()

    def display_board(self, board: chess.Board):
        svg_board = chess.svg.board(
            board,
            size=600,
            lastmove=board.peek() if board.move_stack else None,
            check=board.king(board.turn) if board.is_check() else None
        )
        self.board_placeholder.write(svg_board, unsafe_allow_html=True)

    def display_stats(self, game: ChessGame):
        cols = st.columns(4)
        with cols[0]:
            st.metric("Total Moves", game.stats.total_moves)
        with cols[1]:
            st.metric("Captures", game.stats.capture_moves)
        with cols[2]:
            st.metric("Checks", game.stats.check_moves)
        with cols[3]:
            elapsed = int(time.time() - game.start_time)
            st.metric("Time Elapsed", f"{elapsed}s")

    def display_moves(self, moves: List[GameMove]):
        if not moves:
            return
            
        df = pd.DataFrame([{
            "Move #": i + 1,
            "Player": move.player,
            "Move": move.move,
            "Time": datetime.fromtimestamp(move.timestamp).strftime('%H:%M:%S'),
            "Capture": "✓" if move.is_capture else "",
            "Check": "✓" if move.is_check else ""
        } for i, move in enumerate(moves)])
        
        self.move_history_placeholder.dataframe(df, hide_index=True)

    def create_ai_player(model_type: ModelType) -> ChessAI:
        if model_type in [ModelType.CHATGPT_4O]:
            return OpenAIChessAI(model_type)
        elif model_type == ModelType.CLAUDE_SONNET:
            return AnthropicChessAI(model_type)
        else:
            return LlamaChessAI(model_type)

def main():
    if 'game_ui' not in st.session_state:
        st.session_state.game_ui = ChessUI()
    st.title("🤖 AI Chess Battle")
    st.write("Watch AI models compete in chess!")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("White Player")
        player1_model = st.selectbox(
            "Select Model", 
            options=[m.value for m in ModelType],
            key="white"
        )
    with col2:
        st.subheader("Black Player")
        player2_model = st.selectbox(
            "Select Model", 
            options=[m.value for m in ModelType],
            key="black",
            index=1
        )
    
    if st.button("Start New Game", type="primary"):
        try:
            with st.spinner("Initializing game..."):
                player1 = ChessUI.create_ai_player(ModelType(player1_model))
                player2 = ChessUI.create_ai_player(ModelType(player2_model))
                game = ChessGame(player1, player2)
                ui = ChessUI()
            
            st.success(f"Game started: {player1.name} (White) vs {player2.name} (Black)")
            
            while not game.is_game_over:
                current_player = player1 if len(game.moves) % 2 == 0 else player2
                with st.spinner(f"Thinking... {current_player.name}'s turn"):
                    move = game.play_move()
                    if move:
                        ui.display_board(game.board)
                        ui.display_stats(game)
                        ui.display_moves(game.moves)
                        time.sleep(0.5)
            
            st.balloons()
            st.success(f"Game Over! Winner: {game.stats.winner}")
            ui.display_stats(game)
            ui.display_moves(game.moves)
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Please check your API keys and model configurations.")

if __name__ == "__main__":
    main()