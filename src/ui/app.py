"""
Streamlit UI module for web-based prediction interface.

This module provides a browser-based interface for the next-word predictor using Streamlit.
It is optional and counts as extra credit (+5 points).
"""


class PredictorUI:
    """
    Streamlit-based web UI for next-word prediction.
    
    Provides a user-friendly interface for interacting with the predictor.
    This is an optional extra credit module.
    """

    def __init__(self, predictor):
        """
        Initialize PredictorUI with a Predictor instance.
        
        Args:
            predictor: Pre-loaded Predictor instance.
        """
        pass

    def get_predictions(self, text: str, k: int) -> list:
        """
        Get predictions for display in the UI.
        
        Args:
            text: Input text from user.
            k: Number of predictions to return.
            
        Returns:
            List of predicted words.
        """
        pass

    def run(self):
        """
        Run the Streamlit application.
        
        Launches the web interface for interactive prediction.
        """
        pass


def main():
    """
    Entry point for PredictorUI module.
    
    Launches the Streamlit application.
    """
    ui = PredictorUI(None)
    ui.run()


if __name__ == "__main__":
    main()
