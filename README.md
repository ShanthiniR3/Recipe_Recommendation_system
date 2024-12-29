
# Recipe Recommendation System
This project is a **Recipe Recommender App** built using Python and Streamlit. It recommends recipes based on user preferences such as tags, ingredients, diet restrictions, and minimum ratings.

## Features
1. Allows users to input preferences such as tags, ingredients, dietary restrictions, and minimum ratings.
2. Uses **TF-IDF Vectorization** for text feature extraction.
3. Employs **Cosine Similarity** to find similar recipes.
4. Interactive UI built with **Streamlit**.

## Setup Instructions

### Prerequisites
  Python 3.8 or above and Pip (Python package manager)

## Dataset Details
The dataset should contain:
- **Tags:** Descriptive tags (e.g., "quick, dinner").
- **Ingredients:** List of ingredients.
- **Diet Restrictions:** Information like "gluten-free, vegan."
- **Rating:** Numeric ratings (0 to 5).

---

## Technologies Used
- **Python** for programming
- **Streamlit** for the user interface
- **pandas** and **scikit-learn** for data processing and feature extraction

## How It Works
1. Combines text features (tags, ingredients, dietary restrictions) into a single column.
2. Converts text into numerical vectors using TF-IDF.
3. Computes recipe similarity using Cosine Similarity.
4. Filters and ranks recipes based on user inputs.


## Example Inputs
- Tags: "easy, quick"
- Ingredients: "chicken, garlic"
- Diet Restrictions: "gluten-free"
- Minimum Rating: 4.5

