from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def get_models():
    return {
        "logistic": LogisticRegression(
            max_iter=1000,
            C=0.3,
            penalty="l2"
        ),
        "xgboost": XGBClassifier(
            eval_metric="logloss",
            max_depth=3,              
            learning_rate=0.03,       
            n_estimators=600,         
            subsample=0.8,            
            colsample_bytree=0.8,     
            reg_alpha=2,              
            reg_lambda=3,             
            min_child_weight=10,      
            gamma=1,                  
            random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=5,               
            min_samples_leaf=25,       
            min_samples_split=50,      
            max_features="sqrt",       
            bootstrap=True,            
            random_state=42,
            n_jobs=-1
        ),
        "gbm_sklearn": GradientBoostingClassifier(
            n_estimators=400,          
            learning_rate=0.03,        
            max_depth=3,               
            min_samples_leaf=25,       
            subsample=0.8,             
            max_features="sqrt",       
            validation_fraction=0.1,   
            n_iter_no_change=10,       
            random_state=42
        )
    }