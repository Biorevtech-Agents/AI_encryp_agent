"""
News sentiment analysis module.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any

import nltk
import pandas as pd
from newsapi import NewsApiClient
from textblob import TextBlob
from transformers import pipeline
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download required NLTK data
nltk.download('vader_lexicon')

class SentimentAnalyzer:
    """
    A class for analyzing sentiment from news articles and social media.
    """
    
    def __init__(self, news_api_key: Optional[str] = None):
        """
        Initialize the sentiment analyzer.
        
        Args:
            news_api_key: API key for NewsAPI
        """
        self.news_api_key = news_api_key or os.getenv('NEWS_API_KEY')
        if not self.news_api_key:
            raise ValueError("NewsAPI key is required")
            
        self.news_client = NewsApiClient(api_key=self.news_api_key)
        self.vader = SentimentIntensityAnalyzer()
        self.transformer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        
        self.logger = logging.getLogger(__name__)
        
    def get_news_articles(
        self,
        query: str,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        language: str = 'en',
        sort_by: str = 'publishedAt'
    ) -> List[Dict[str, Any]]:
        """
        Fetch news articles from NewsAPI.
        
        Args:
            query: Search query
            from_date: Start date for articles
            to_date: End date for articles
            language: Language of articles
            sort_by: Sorting criteria
            
        Returns:
            List of news articles
        """
        try:
            # Set default dates if not provided
            if not to_date:
                to_date = datetime.now()
            if not from_date:
                from_date = to_date - timedelta(days=7)
                
            # Format dates for NewsAPI
            from_str = from_date.strftime('%Y-%m-%d')
            to_str = to_date.strftime('%Y-%m-%d')
            
            # Get articles
            response = self.news_client.get_everything(
                q=query,
                from_param=from_str,
                to=to_str,
                language=language,
                sort_by=sort_by
            )
            
            return response['articles']
            
        except Exception as e:
            self.logger.error(f"Error fetching news articles: {str(e)}")
            return []
            
    def analyze_text_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of a text using multiple methods.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        try:
            # VADER sentiment
            vader_scores = self.vader.polarity_scores(text)
            
            # TextBlob sentiment
            blob = TextBlob(text)
            textblob_polarity = blob.sentiment.polarity
            textblob_subjectivity = blob.sentiment.subjectivity
            
            # Transformer sentiment
            transformer_result = self.transformer(text)[0]
            transformer_sentiment = 1 if transformer_result['label'] == 'POSITIVE' else 0
            transformer_score = transformer_result['score']
            
            # Combine scores
            return {
                'vader_compound': vader_scores['compound'],
                'vader_pos': vader_scores['pos'],
                'vader_neg': vader_scores['neg'],
                'vader_neu': vader_scores['neu'],
                'textblob_polarity': textblob_polarity,
                'textblob_subjectivity': textblob_subjectivity,
                'transformer_sentiment': transformer_sentiment,
                'transformer_score': transformer_score
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing text sentiment: {str(e)}")
            return {}
            
    def analyze_articles_sentiment(
        self,
        articles: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Analyze sentiment for a list of news articles.
        
        Args:
            articles: List of news articles
            
        Returns:
            DataFrame with sentiment analysis results
        """
        results = []
        
        for article in articles:
            try:
                # Extract article text
                title = article.get('title', '')
                description = article.get('description', '')
                content = article.get('content', '')
                
                # Combine text fields
                text = ' '.join(filter(None, [title, description, content]))
                
                if not text:
                    continue
                    
                # Get sentiment scores
                sentiment_scores = self.analyze_text_sentiment(text)
                
                # Add article metadata
                result = {
                    'title': title,
                    'published_at': article.get('publishedAt'),
                    'source': article.get('source', {}).get('name'),
                    'url': article.get('url'),
                    **sentiment_scores
                }
                
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Error analyzing article: {str(e)}")
                continue
                
        # Create DataFrame
        df = pd.DataFrame(results)
        if not df.empty:
            df['published_at'] = pd.to_datetime(df['published_at'])
            df = df.set_index('published_at')
            
        return df
        
    def get_aggregated_sentiment(
        self,
        query: str,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        resample_freq: str = '1D'
    ) -> pd.DataFrame:
        """
        Get aggregated sentiment scores for a query over time.
        
        Args:
            query: Search query
            from_date: Start date
            to_date: End date
            resample_freq: Frequency for resampling data
            
        Returns:
            DataFrame with aggregated sentiment scores
        """
        # Get articles
        articles = self.get_news_articles(query, from_date, to_date)
        
        if not articles:
            return pd.DataFrame()
            
        # Analyze sentiment
        df = self.analyze_articles_sentiment(articles)
        
        if df.empty:
            return df
            
        # Resample and aggregate
        agg_dict = {
            'vader_compound': 'mean',
            'vader_pos': 'mean',
            'vader_neg': 'mean',
            'vader_neu': 'mean',
            'textblob_polarity': 'mean',
            'textblob_subjectivity': 'mean',
            'transformer_sentiment': 'mean',
            'transformer_score': 'mean',
            'title': 'count'
        }
        
        df_resampled = df.resample(resample_freq).agg(agg_dict)
        df_resampled = df_resampled.rename(columns={'title': 'article_count'})
        
        return df_resampled 