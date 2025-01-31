1. Overview:
This data set provides a detailed view of player behavior and demographics in online gaming environments. It includes a variety of features that capture key aspects of gaming activity, player characteristics and engagemnt levels, making it an excellent ressource for analyzing player retention and predicting engagement patterns.
2. Features:
- Player Demographics 🧍‍♂️🧍‍♀️:
PlayerID 🆔: Unique identifier for each player.
Age 🎂: Age of the player.
Gender ⚧️: Gender of the player.
Location 🌍: Geographic location of the player.
- Game-Specific Attributes 🎮:
GameGenre 🎭: Genre of the game the player is engaged in (e.g., RPG, Strategy, Action).
GameDifficulty 🎯: Difficulty level of the game, categorized as Easy, Medium, or Hard.
PlayerLevel 📈: Current level of the player in the game.
AchievementsUnlocked 🏆: Number of achievements unlocked by the player.
- Engagement Metrics 📊:
PlayTimeHours ⏳: Average hours spent playing per session.
InGamePurchases 💰: Indicates whether the player makes in-game purchases (0 = No, 1 = Yes).
SessionsPerWeek 📅: Number of gaming sessions per week.
AvgSessionDurationMinutes ⏱️: Average duration of each gaming session in minutes.
- Target Variable 🎯:
EngagementLevel 🚀: The target variable categorizes player engagement levels into three classes: 'High', 'Medium', and 'Low', reflecting player retention and activity.
3. Data Highlights:
- The data includes 40,034 player entries with 13 features
- Features are a mix pf numerical and categorical data
- no missing values are present ✅
4. Objective 🎯:
The primary goal is to analyze and predict player engagement levels based on their gaming behavior, preferences, and demographics. This analysis aims at:
- Identify factors influencing engagement 🔑:
* Pinpoint the key features that drive higher engagement levels such as session duration, player level, and frequency of gaming
- Develop predictive models 🤖:
* Build machine learning models to classify players into engagement levels (Low, Medium, High) with high accuracy and interpretability
- Provide actionable insights 💡:
* Insights for game developers and marketers to enhance player retention and improve overall gaming experience
- Optimize engagement strategies 🧩:
* Suggest interventions such as tailored rewards, challenges, or personalized recommendations to maximize player engagement and retention
5. Machine Learnig models results:
![Machine Learning models results](../prediction/model_results.csv)
Cat boost was the best performing model for predicting player retention (92% accuracy)
6. Feature importance for best model:
![Machine Learning models results](../figures/best%20model%20feature%20importance.png)