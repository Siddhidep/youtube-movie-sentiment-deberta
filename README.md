# youtube-movie-sentiment-deberta
🎬 YouTube Movie Review Sentiment Analysis with DeBERTa

A comprehensive sentiment analysis pipeline that scrapes YouTube movie review comments, performs data cleaning and preprocessing, and trains a DeBERTa transformer model for 6-class sentiment classification (0-5 scale). Features advanced persona clustering to categorize viewers as Critics, Casual Viewers, or Superfans based on comment patterns and sentiment scores.

🔧 Key Features:
- Multi-threaded YouTube comment scraping from movie videos
- Advanced text preprocessing with emoji removal, language detection, and meaningfulness filtering  
- Fine-tuned microsoft/deberta-v3-small model with class balancing
- Persona clustering using DeBERTa embeddings + K-means
- Interactive visualization with Plotly
- Real-time sentiment prediction for new comments

📊 Tech Stack: Python, PyTorch, Transformers (Hugging Face), NLTK, scikit-learn, Plotly

🎯 Achieves ~75% accuracy on movie review sentiment classification with robust handling of class imbalance through weighted loss functions.
