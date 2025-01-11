from flask import Flask, render_template, request
import pickle
import numpy as np

# Load preprocessed data and models
popular_df = pickle.load(open('popular.pkl','rb'))
pt = pickle.load(open('pt.pkl','rb'))
books = pickle.load(open('books.pkl','rb'))
similarity_scores = pickle.load(open('similarity_scores.pkl','rb'))

# Initialize the Flask application
app = Flask(__name__)

# Home route
@app.route('/')
def index():

    """
    Renders the home page with popular books and their details.

    """
    return render_template('index.html',
                           book_name = list(popular_df['Book-Title'].values),
                           author = list(popular_df['Book-Author'].values),
                           image = list(popular_df['Image-URL-M'].values),
                           votes = list(popular_df['num_ratings'].values),
                           rating = list(popular_df['avg_rating'].values)
                           )


# Recommendation page UI
@app.route('/recommend')
def recommend_ui():
    """
    Renders the recommendation page.
    """
    return render_template('recommend.html')


# Book recommendation logic
@app.route('/recommend_books',methods=['post'])
def recommend():

    """
    Processes the user input and provides book recommendations.
    """
    user_input = request.form.get('user_input')
    # Find the index of the user input in the pivot table

    try:
        index = np.where(pt.index == user_input)[0][0]

        # Get the top 4 similar items
        similar_items = sorted(list(enumerate(similarity_scores[index])), key = lambda x: x[1], reverse = True)[1:5]

        data = []
        for i in similar_items:
            item = []
            temp_df = books[books['Book-Title'] == pt.index[i[0]]]
            item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
            item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
            item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))

            data.append(item)

        print(data)
        return render_template('recommend.html', data = data)
    
    except IndexError:
        # Handle cases where the user input is not found
        error_message = "The book you entered is not in our database. Please try another one."
        
        return render_template('recommend.html', error = error_message)

# Run the Flask application
if __name__ == '__main__':
    app.run(debug = True)