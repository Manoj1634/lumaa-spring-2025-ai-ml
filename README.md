
# Recipe Recommendation System

Link to the video
https://drive.google.com/file/d/13NzvJ7mgq-WJnlFqBh_mMNz52SoadkWi/view?usp=drive_link

This project is a Recipe Recommendation System that allows users to find recipes based on specific ingredients they want to include or avoid. It utilizes advanced text processing techniques and a machine learning model to suggest recipes that closely match the user's preferences.

## Features

- **Ingredient-based Filtering:** Users can specify ingredients they want to include or exclude from their recipes.
- **Dietary Preferences:** Options to exclude meat or dairy products.
- **Rating and Time Filters:** Users can set minimum ratings and maximum cooking times for tailored recommendations.
- **User-Friendly Interface:** Built with Streamlit, the application provides an easy-to-use interface for all user interactions.

## Installation

To run this project, you will need Python and several libraries installed. Here are the steps to set it up:

### Prerequisites

- Python 3.x
- pip
- virtualenv (optional)

### Clone the Repository

First, clone this repository to your local machine:

```bash
git clone https://github.com/yourusername/recipe-recommendation-system.git
cd recipe-recommendation-system
```
### Install Dependencies

Install all required packages using `pip`:
```bash  
pip install -r requirements.txt
```

## Usage

To run the application, execute the following command from the root directory of the project:
```bash
streamlit run new_app.py
```


Navigate to `http://localhost:8501` in your web browser to view the application.

## Data

Used the dataset from below and Feature Engineered it

https://www.kaggle.com/datasets/thedevastator/better-recipes-for-a-better-life?select=recipes.csv
The system uses a preprocessed dataset of recipes which include various attributes like ingredients, cooking time, and nutritional information. The dataset is located under `data/Processed_df.csv`. The preprocessing is performed in `preprocessing.py`, where data is transformed and cleaned to suit the needs of the recommendation engine.


## Application

![Application Screenshot](https://i.imgur.com/zPkVAnA.png)


## Query Options
 For changing the query preferance make the changes as below in new_app.py
- **By Ingredients Only:** Set the query to use the `processed_ingredients` column for recommendations.
- **By Ingredients and Directions:** Use the `processed_ingredients_directions` column to incorporate both ingredients and cooking directions in your search.
- **By Ingredients and Quantity:** For queries that consider the quantity of ingredients, use the `ingredients` column.

## More Information
for more info visit Report_mj.pdf

## Contributing

Contributions to this project are welcome. You can contribute in several ways:

- Adding new features
- Improving the algorithm
- Enhancing the user interface
- Fixing bugs or issues

Please feel free to fork the repository and submit pull requests.


