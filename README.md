# Titanic Survival Prediction

Trains a **RandomForest classifier** on essential passenger features (**Pclass**, **Sex**, **SibSp**, **Parch**). After simple encoding using `get_dummies` and training with **100 trees** and **max_depth=5**, it achieves a **0.77511** score on the Kaggle public leaderboard.

## How It Works:
1. **Load the Titanic dataset** from Kaggle.  
   Dataset link: [Titanic - Machine Learning from Disaster](https://www.kaggle.com/c/titanic)
2. Perform **simple preprocessing**, including encoding categorical variables with `get_dummies`.
3. Train a **RandomForest classifier** with:
   - 100 trees
   - max_depth=5
4. **Evaluate the model** and submit predictions to the Kaggle competition, achieving a score of **0.77511** on the public leaderboard.
