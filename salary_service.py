# salary_service.py
import os
import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import time
import json
from datetime import datetime
import threading
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
retraining_progress = {}
progress_lock = threading.Lock()




# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def set_retraining_progress(session_id, percentage, stage, step_id=None, error=None):
    """Set retraining progress for a session"""
    with progress_lock:
        retraining_progress[session_id] = {
            'percentage': percentage,
            'stage': stage,
            'step_id': step_id,
            'error': error,
            'timestamp': datetime.now().isoformat(),
            'completed': percentage >= 100,
            'failed': error is not None
        }

def get_retraining_progress(session_id):
    """Get retraining progress for a session"""
    with progress_lock:
        return retraining_progress.get(session_id, {
            'percentage': 0,
            'stage': 'Not started',
            'step_id': None,
            'error': None,
            'timestamp': datetime.now().isoformat(),
            'completed': False,
            'failed': False
        })
    
def clear_retraining_progress(session_id):
    """Clear progress data for a session"""
    with progress_lock:
        if session_id in retraining_progress:
            del retraining_progress[session_id]

class SalaryDataCollector:
    """Phase 1: Data Collection and Quality Assessment with Progress Tracking"""
    
    def __init__(self, api_base_url=None, api_key=None, host=None, session_id=None):
        self.api_base_url = api_base_url or os.getenv("SALARY_API_URL")
        self.api_key = api_key or os.getenv("SALARY_API_KEY")
        self.host = host or os.getenv("HOST")
        self.quality_threshold = {"min_sample_size": 5, "required_confidence": ["MEDIUM", "HIGH", "VERY_HIGH"]}
        self.session_id = session_id
        
    def update_progress(self, percentage, stage, step_id=None):
        """Update progress if session_id is provided"""
        if self.session_id:
            set_retraining_progress(self.session_id, percentage, stage, step_id)
            logger.info(f"Progress: {percentage}% - {stage}")
    
    def make_api_call(self, job_title, location, experience_level, location_type="CITY", max_retries=3):
        """Make API call with error handling and retry logic for rate limits"""
        for attempt in range(max_retries):
            try:
                params = {
                    "job_title": job_title,
                    "location": location,
                    "location_type": location_type,
                    "years_of_experience": experience_level
                }
                
                headers = {
                    "x-rapidapi-key": self.api_key,
                    "x-rapidapi-host": self.host
                }
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"
                
                response = requests.get(self.api_base_url, params=params, headers=headers)
                response.raise_for_status()
                
                return response.json()
                
            except requests.exceptions.HTTPError as e:
                if response.status_code == 429:  # Rate limit
                    wait_time = 2 ** attempt * 5  # Exponential backoff: 5, 10, 20 seconds
                    logger.warning(f"Rate limit hit for {job_title}, {location}. Waiting {wait_time}s (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"Rate limit exceeded after {max_retries} attempts for {job_title}, {location}, {experience_level}")
                        return None
                else:
                    logger.error(f"HTTP error {response.status_code} for {job_title}, {location}, {experience_level}: {e}")
                    return None
            except requests.exceptions.RequestException as e:
                logger.error(f"API call failed for {job_title}, {location}, {experience_level}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Simple backoff for other errors
                    continue
                return None
        
        return None

    def assess_data_quality(self, api_response):
        """Assess if API response meets quality standards"""
        if not api_response or api_response.get("status") != "OK":
            return False, "API call failed"
        
        data = api_response.get("data", [])
        if not data:
            return False, "No data returned"
        
        sample_data = data[0]  # Assuming single response
        sample_size = sample_data.get("salary_count", 0)
        confidence = sample_data.get("confidence", "LOW")
        
        quality_score = {
            "sample_size": sample_size,
            "confidence": confidence,
            "meets_threshold": (
                sample_size >= self.quality_threshold["min_sample_size"] and 
                confidence in self.quality_threshold["required_confidence"]
            )
        }
        
        return quality_score["meets_threshold"], quality_score
    
    def extract_data_point(self, api_response, job_title, location, experience):
        """Extract clean data point from API response"""
        try:
            data = api_response["data"][0]
            
            return {
                "job_title": job_title,
                "location": location,  
                "experience_level": experience,
                "min_salary": data.get("min_salary"),
                "max_salary": data.get("max_salary"),
                "median_salary": data.get("median_salary"),
                "min_base_salary": data.get("min_base_salary"),
                "max_base_salary": data.get("max_base_salary"), 
                "median_base_salary": data.get("median_base_salary"),
                "salary_period": data.get("salary_period", "MONTH"),
                "salary_currency": data.get("salary_currency"),
                "salary_count": data.get("salary_count"),
                "confidence": data.get("confidence"),
                "collected_at": datetime.now().isoformat()
            }
        except (KeyError, IndexError) as e:
            logger.error(f"Error extracting data point: {e}")
            return None
    
    def collect_strategic_data(self, batch_size=5):
        """Collect data for strategic job/location/experience combinations with progress tracking"""
        
        # Define strategic combinations based on common HR needs
        priority_combinations = [
            # Data science roles
            ("data scientist", "cape town", "LESS_THAN_ONE"),
            ("data scientist", "johannesburg", "LESS_THAN_ONE"),
            ("data scientist", "cape town", "ONE_TO_THREE"),
            ("data scientist", "johannesburg", "ONE_TO_THREE"),
            ("data scientist", "cape town", "FOUR_TO_SIX"),
            ("data scientist", "johannesburg", "FOUR_TO_SIX"),
            ("data analyst", "cape town", "LESS_THAN_ONE"),
            ("data scientist", "cape town", "SEVEN_TO_NINE"),
            ("data scientist", "johannesburg", "SEVEN_TO_NINE"),

            # Data analyst roles
            ("data analyst", "cape town", "LESS_THAN_ONE"),
            ("data analyst", "johannesburg", "LESS_THAN_ONE"),
            ("data analyst", "cape town", "ONE_TO_THREE"),
            ("data analyst", "johannesburg", "ONE_TO_THREE"),
            ("data analyst", "cape town", "FOUR_TO_SIX"),
            ("data analyst", "johannesburg", "FOUR_TO_SIX"),
            ("data analyst", "cape town", "SEVEN_TO_NINE"),
            ("data analyst", "johannesburg", "SEVEN_TO_NINE"),

            # Software roles
            ("software developer", "cape town", "LESS_THAN_ONE"),
            ("software developer", "johannesburg", "LESS_THAN_ONE"),
            ("software engineer", "cape town", "ONE_TO_THREE"),
            ("software engineer", "johannesburg", "ONE_TO_THREE"),
            ("software engineer", "cape town", "FOUR_TO_SIX"),
            ("software engineer", "johannesburg", "FOUR_TO_SIX"),
            ("software engineer", "cape town", "SEVEN_TO_NINE"),
            ("software engineer", "johannesburg", "SEVEN_TO_NINE"),
            ("full stack developer", "johannesburg", "ONE_TO_THREE"),
            ("backend developer", "johannesburg", "FOUR_TO_SIX"),
            
            # Other tech roles
            ("devops engineer", "cape town", "FOUR_TO_SIX"),
            ("machine learning engineer", "johannesburg", "FOUR_TO_SIX"),
            ("product manager", "cape town", "FOUR_TO_SIX"),
            ("ui/ux designer", "cape town", "ONE_TO_THREE"),
        ]
        
        collected_data = []
        quality_report = {}
        failed_requests = 0
        max_failures = 10  # Stop if too many failures
        
        total_combinations = len(priority_combinations)
        
        # Update initial progress
        self.update_progress(5, 'Starting data collection...', 'step-collect')
        
        logger.info(f"Starting data collection for {total_combinations} combinations in batches of {batch_size}...")
        
        for i, (job_title, location, experience) in enumerate(priority_combinations):
            # Calculate progress (5% to 85% range for data collection)
            base_progress = 5
            collection_progress_range = 80  # 85 - 5 = 80%
            current_progress = base_progress + int((i / total_combinations) * collection_progress_range)
            
            self.update_progress(
                current_progress, 
                f'Collecting data {i+1}/{total_combinations}: {job_title} in {location}',
                'step-collect'
            )
            
            logger.info(f"Collecting data {i+1}/{total_combinations}: {job_title} in {location}")
            
            # Stop if too many failures (likely rate limit issues)
            if failed_requests >= max_failures:
                logger.warning(f"Too many failures ({failed_requests}). Stopping data collection.")
                break
            
            # Make API call
            api_response = self.make_api_call(job_title, location, experience)
            
            if api_response:
                # Reset failure counter on success
                failed_requests = 0
                
                # Assess quality
                is_quality, quality_info = self.assess_data_quality(api_response)
                quality_report[(job_title, location, experience)] = quality_info
                
                if is_quality:
                    # Extract and clean data
                    data_point = self.extract_data_point(api_response, job_title, location, experience)
                    if data_point:
                        collected_data.append(data_point)
                        logger.info(f"✓ High quality data collected: {quality_info['sample_size']} samples, {quality_info['confidence']} confidence")
                else:
                    logger.warning(f"✗ Low quality data: {quality_info}")
            else:
                failed_requests += 1
                logger.warning(f"Failed to collect data (failure {failed_requests}/{max_failures})")
            
            # Respect API rate limits with longer delay
            time.sleep(3)  # 3 seconds between requests to avoid rate limiting
            
            # Take a longer break after each batch
            if (i + 1) % batch_size == 0:
                batch_num = (i + 1) // batch_size
                self.update_progress(
                    current_progress,
                    f'Completed batch {batch_num}. Taking break...',
                    'step-collect'
                )
                logger.info(f"Completed batch {batch_num}. Taking 10 second break...")
                time.sleep(10)
        
        # Final data collection progress
        self.update_progress(85, f'Data collection complete: {len(collected_data)} high-quality data points', 'step-collect')
        logger.info(f"Data collection complete: {len(collected_data)} high-quality data points collected, {failed_requests} failures")
        return collected_data, quality_report

class SalaryPredictor:
    """Phase 1: Baseline ML Model"""
    
    def __init__(self, model_path="salary_model.joblib"):
        self.model = None
        self.label_encoders = {}
        self.is_trained = False
        self.feature_columns = ['job_title', 'location', 'experience_level']
        self.model_path = model_path
        
        # Try to load existing model on initialization
        self._try_load_model()
        
    def _try_load_model(self):
        """Try to load an existing model on initialization"""
        if os.path.exists(self.model_path):
            try:
                self.load_model(self.model_path)
                logger.info("Existing salary model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load existing model: {e}")
                
    def prepare_features(self, df):
        """Encode categorical features for ML model"""
        df_encoded = df.copy()
        
        for column in self.feature_columns:
            if column not in self.label_encoders:
                self.label_encoders[column] = LabelEncoder()
                df_encoded[f'{column}_encoded'] = self.label_encoders[column].fit_transform(df[column])
            else:
                # Handle unseen categories during prediction
                le = self.label_encoders[column]
                mask = df[column].isin(le.classes_)
                df_encoded[f'{column}_encoded'] = -1  # Default for unseen categories
                df_encoded.loc[mask, f'{column}_encoded'] = le.transform(df.loc[mask, column])
        
        return df_encoded
    
    def train_model(self, training_data):
        """Train the baseline RandomForest model"""
        df = pd.DataFrame(training_data)
        
        if len(df) < 10:
            raise ValueError("Insufficient training data. Need at least 10 data points.")
        
        # Clean and validate salary data
        df = df.dropna(subset=['median_salary'])
        df = df[df['median_salary'] > 0]  # Remove invalid salaries
        
        if len(df) < 5:
            raise ValueError("Insufficient valid training data after cleaning.")
        
        # Normalize salary period (convert monthly to annual)
        df['annual_median_salary'] = df.apply(lambda row: 
            row['median_salary'] * 12 if row['salary_period'] == 'MONTH' 
            else row['median_salary'], axis=1)
        
        # Remove outliers (salaries beyond reasonable range)
        Q1 = df['annual_median_salary'].quantile(0.25)
        Q3 = df['annual_median_salary'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df = df[(df['annual_median_salary'] >= lower_bound) & 
                (df['annual_median_salary'] <= upper_bound)]
        
        logger.info(f"Training with {len(df)} data points after cleaning")
        
        # Prepare features
        df_encoded = self.prepare_features(df)
        
        # Features and target
        feature_cols = [f'{col}_encoded' for col in self.feature_columns]
        X = df_encoded[feature_cols]
        y = df_encoded['annual_median_salary']
        
        if len(df) < 30:
            # Don't split - use all data for training and testing
            X_train = X_test = X
            y_train = y_test = y
            logger.warning("Small dataset: Using all data for both training and testing")
        else:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        self.model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=2
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        
        evaluation = {
            "mae": mean_absolute_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "r2": r2_score(y_test, y_pred),
            "training_samples": len(df),
            "test_samples": len(y_test)
        }
        
        self.is_trained = True
        logger.info(f"Model trained successfully: R² = {evaluation['r2']:.3f}, MAE = {evaluation['mae']:.0f}")
        
        # Save the model
        self.save_model(self.model_path)
        
        return evaluation
    
    def predict_salary(self, job_title, location, experience_level):
        """Predict salary for given inputs"""
        if not self.is_trained:
            # Try to load the model if it exists
            if not self._try_load_model():
                raise ValueError("Model must be trained before making predictions. No trained model found.")
        
        # Create input DataFrame
        input_data = pd.DataFrame([{
            'job_title': job_title.lower().strip(),
            'location': location.lower().strip(),
            'experience_level': experience_level.upper().strip()
        }])
        
        # Check if we have encoders for all inputs
        for column in self.feature_columns:
            if column not in self.label_encoders:
                raise ValueError(f"Model not trained for feature: {column}")
        
        # Encode features
        input_encoded = self.prepare_features(input_data)
        feature_cols = [f'{col}_encoded' for col in self.feature_columns]
        X_input = input_encoded[feature_cols]
        
        # Check for unseen categories (encoded as -1)
        if (X_input == -1).any().any():
            # Handle unseen categories by using fallback predictions
            logger.warning("Unseen category detected, using fallback prediction")
            return self._fallback_prediction(job_title, location, experience_level)
        
        # Make prediction
        prediction = self.model.predict(X_input)[0]
        
        # Estimate range (±20% of prediction)
        prediction_range = (prediction * 0.8, prediction * 1.2)
        
        # Add market insights
        insights = self._generate_market_insights(job_title, location, experience_level, prediction)
        
        return {
            "predicted_annual_salary": round(prediction),
            "predicted_range": (round(prediction_range[0]), round(prediction_range[1])),
            "market_insights": insights,
            "source": "ML_MODEL"
        }
    
    def _try_load_model(self):
        """Try to load an existing model on initialization"""
        if os.path.exists(self.model_path):
            try:
                self.load_model(self.model_path)
                logger.info("Existing salary model loaded successfully")
                return True
            except Exception as e:
                logger.warning(f"Failed to load existing model: {e}")
                return False
        return False

    
    def _generate_market_insights(self, job_title, location, experience_level, prediction):
        """Generate market insights based on prediction"""
        insights = []
        
        # Experience-based insights
        if "LESS_THAN_ONE" in experience_level:
            insights.append("Entry-level position with growth potential")
        elif "SEVEN_TO_NINE" in experience_level or "TEN_PLUS" in experience_level:
            insights.append("Senior-level role commanding premium salaries")
        
        # Location-based insights
        if "cape town" in location.lower():
            insights.append("Cape Town market shows competitive tech salaries")
        elif "johannesburg" in location.lower():
            insights.append("Johannesburg offers premium for financial and tech sectors")
        
        # Salary range insights
        if prediction > 1000000:
            insights.append("Above-average salary range for this combination")
        elif prediction < 400000:
            insights.append("Entry-level salary range")
        
        return ". ".join(insights) if insights else f"Market data for {job_title} in {location}"
    
    def save_model(self, filepath):
        """Save trained model and encoders"""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        model_data = {
            "model": self.model,
            "label_encoders": self.label_encoders,
            "feature_columns": self.feature_columns,
            "version": "1.0",
            "trained_at": datetime.now().isoformat()
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model and encoders"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
            
        try:
            model_data = joblib.load(filepath)
            
            self.model = model_data["model"]
            self.label_encoders = model_data["label_encoders"]
            self.feature_columns = model_data["feature_columns"]
            self.is_trained = True
            
            logger.info(f"Model loaded from {filepath}")
            
            # Log model info if available
            if "trained_at" in model_data:
                logger.info(f"Model was trained at: {model_data['trained_at']}")
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model from {filepath}: {e}")
            raise

class CompetitivenessAnalyzer:
    """Phase 1: Basic Competitiveness Analysis"""
    
    @staticmethod
    def analyze_competitiveness(predicted_salary, predicted_range, user_budget, confidence="MEDIUM"):
        """Analyze how competitive user's budget is vs market prediction"""
        
        min_market, max_market = predicted_range
        pct_vs_median = (user_budget - predicted_salary) / predicted_salary * 100
        
        # Determine competitiveness level
        if user_budget > max_market:
            level = "highly_competitive"
            flag = f"Budget is {abs(pct_vs_median):.1f}% above market - Highly Competitive"
            color = "green"
            recommendation = "Excellent positioning to attract top talent"
            
        elif user_budget >= predicted_salary:
            level = "competitive" 
            flag = f"Budget is {abs(pct_vs_median):.1f}% above median - Competitive"
            color = "blue"
            recommendation = "Good positioning for quality candidates"
            
        elif user_budget >= min_market:
            level = "somewhat_competitive"
            flag = f"Budget is {abs(pct_vs_median):.1f}% below median but within market range"
            color = "yellow"
            recommendation = "May attract junior or motivated candidates"
            
        else:
            level = "not_competitive"
            flag = f"Budget is {abs(pct_vs_median):.1f}% below market - Not Competitive" 
            color = "red"
            recommendation = "Consider increasing budget or adjusting requirements"
        
        return {
            "level": level,
            "flag": flag,
            "color": color,
            "pct_vs_median": round(pct_vs_median, 1),
            "recommendation": recommendation,
            "market_position": {
                "user_budget": user_budget,
                "market_median": predicted_salary,
                "market_range": predicted_range,
                "within_range": min_market <= user_budget <= max_market
            },
            "confidence": confidence
        }

# Global instances
salary_collector = None
salary_predictor = None
competitiveness_analyzer = CompetitivenessAnalyzer()

def initialize_salary_predictor_with_progress(session_id, force_retrain=False):
    """Initialize the salary predictor with progress tracking"""
    global salary_collector, salary_predictor
    
    try:
        set_retraining_progress(session_id, 2, 'Initializing salary predictor...', 'step-init')
        
        # Initialize instances with session_id for progress tracking
        salary_collector = SalaryDataCollector(session_id=session_id)
        salary_predictor = SalaryPredictor()
        
        set_retraining_progress(session_id, 5, 'Initialized data collector and predictor', 'step-init')
        
        # If force_retrain is False, check if model is already loaded
        if not force_retrain and salary_predictor.is_trained:
            set_retraining_progress(session_id, 100, 'Salary predictor model already loaded and ready', 'step-save')
            logger.info("Salary predictor model already loaded and ready")
            return True
        
        # If force_retrain is False, try to load existing model first
        if not force_retrain and os.path.exists(salary_predictor.model_path):
            try:
                salary_predictor.load_model(salary_predictor.model_path)
                set_retraining_progress(session_id, 100, 'Existing model loaded successfully', 'step-save')
                logger.info("Salary predictor model loaded successfully from file")
                return True
            except Exception as e:
                logger.warning(f"Failed to load existing model: {e}")
                set_retraining_progress(session_id, 8, 'Failed to load existing model, collecting fresh data...', 'step-collect')
        
        # Collect fresh data and train model
        if force_retrain:
            set_retraining_progress(session_id, 8, 'Force retrain requested. Collecting fresh data...', 'step-collect')
            logger.info("Force retrain requested. Collecting fresh data...")
        else:
            set_retraining_progress(session_id, 8, 'No existing model found. Collecting data...', 'step-collect')
            logger.info("No existing model found. Collecting data to train new model...")
            
        # This is where training_data gets defined - with progress tracking
        training_data, quality_report = salary_collector.collect_strategic_data()
        
        if len(training_data) >= 10:
            set_retraining_progress(session_id, 87, 'Data collection complete. Preparing for model training...', 'step-quality')
            time.sleep(1)
            
            set_retraining_progress(session_id, 90, 'Starting model training...', 'step-train')
            evaluation = salary_predictor.train_model(training_data)
            
            set_retraining_progress(session_id, 95, 'Model training complete. Saving model...', 'step-save')
            time.sleep(1)
            
            action = "retrained" if force_retrain else "trained"
            set_retraining_progress(session_id, 100, f'Model {action} successfully!', 'step-save')
            logger.info(f"Salary predictor {action} successfully: R² = {evaluation['r2']:.3f}")
            return True
        else:
            set_retraining_progress(session_id, 0, 'Insufficient training data collected', None, 'Not enough quality data points collected from API')
            logger.warning("Insufficient training data collected")
            return False
            
    except Exception as e:
        error_msg = f"Failed to initialize salary predictor: {e}"
        set_retraining_progress(session_id, 0, 'Initialization failed', None, error_msg)
        logger.error(error_msg)
        return False

def retrain_salary_model_with_progress(session_id):
    """Force retrain the salary prediction model with progress tracking"""
    logger.info("Starting forced model retraining with progress tracking...")
    return initialize_salary_predictor_with_progress(session_id, force_retrain=True)


def initialize_salary_predictor(force_retrain=False):
    """Initialize the salary predictor by training or loading the model
    
    Args:
        force_retrain (bool): If True, forces fresh data collection and retraining
                             even if a model already exists
    """
    global salary_collector, salary_predictor
    
    try:
        # Initialize instances
        salary_collector = SalaryDataCollector()
        salary_predictor = SalaryPredictor()
        
        # If force_retrain is False, check if model is already loaded
        if not force_retrain and salary_predictor.is_trained:
            logger.info("Salary predictor model already loaded and ready")
            return True
        
        # If force_retrain is False, try to load existing model first
        if not force_retrain and os.path.exists(salary_predictor.model_path):
            try:
                salary_predictor.load_model(salary_predictor.model_path)
                logger.info("Salary predictor model loaded successfully from file")
                return True
            except Exception as e:
                logger.warning(f"Failed to load existing model: {e}")
                logger.info("Will collect data and train new model...")
        
        # Collect fresh data and train model
        if force_retrain:
            logger.info("Force retrain requested. Collecting fresh data...")
        else:
            logger.info("No existing model found. Collecting data to train new model...")
            
        # This is where training_data gets defined
        training_data, quality_report = salary_collector.collect_strategic_data()
            
        if len(training_data) >= 10:
            evaluation = salary_predictor.train_model(training_data)
            action = "retrained" if force_retrain else "trained"
            logger.info(f"Salary predictor {action} successfully: R² = {evaluation['r2']:.3f}")
            return True
        else:
            logger.warning("Insufficient training data collected")
            return False
            
    except Exception as e:
        logger.error(f"Failed to initialize salary predictor: {e}")
        return False


def retrain_salary_model():
    """Force retrain the salary prediction model with fresh data
    
    This function is specifically designed for the retrain API endpoint
    """
    logger.info("Starting forced model retraining...")
    return initialize_salary_predictor(force_retrain=True)

def get_salary_predictor():
    """Get the global salary predictor instance, initializing if necessary"""
    global salary_predictor
    
    if salary_predictor is None or not salary_predictor.is_trained:
        logger.info("Salary predictor not initialized. Initializing now...")
        if not initialize_salary_predictor():
            raise RuntimeError("Failed to initialize salary predictor")
    
    return salary_predictor

def get_competitiveness_analyzer():
    """Get the global competitiveness analyzer instance"""
    return competitiveness_analyzer

# Initialize the predictor when the module is imported (but don't force retrain)

def _safe_initialize():
    """Safely initialize salary predictor without raising exceptions during import"""
    try:
        return initialize_salary_predictor(force_retrain=False)
    except Exception as e:
        logger.warning(f"Failed to initialize salary predictor on import: {e}")
        logger.info("Salary predictor will be initialized on first use")
        return False

# Initialize when module is imported, but don't fail if it doesn't work
_initialization_success = _safe_initialize()