import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from IPython.display import HTML
from flask import Flask, request, render_template
import ast

app = Flask(__name__)

df = pd.read_csv('/Users/Ricky/Desktop/UTS sem 5/AI/out.csv')
df.astype({'name': str, 'tags': str})


df['tags'] = df['tags'].str.replace(r"\[", "", regex=True)
df['tags'] = df['tags'].str.replace(r"\]", "", regex=True)
df['tags'] = df['tags'].str.replace("'", "", regex=True)
df['steps'] = df['steps'].str.replace(r"\[", "", regex=True)
df['steps'] = df['steps'].str.replace(r"\]", "", regex=True)
df['steps'] = df['steps'].str.replace("'", "", regex=True)
df['ingredients'] = df['ingredients'].str.replace(r"\[", "", regex=True)
df['ingredients'] = df['ingredients'].str.replace(r"\]", "", regex=True)
df['ingredients'] = df['ingredients'].str.replace("'", "", regex=True)
dftest = df.drop(columns=['steps', 'Unnamed: 0', 'minutes', 'id', 'contributor_id',
                 'submitted', 'nutrition', 'description', 'ingredients', 'n_steps'])


@app.route('/')
def index():
    # menampilkan template
    return render_template('index.html')


@app.route('/prediction', methods=['POST'])
def prediction():
    african = str(request.form['african'])
    american = str(request.form['american'])
    asian = str(request.form['asian'])
    australian = str(request.form['australian'])
    chinese = str(request.form['chinese'])
    european = str(request.form['european'])
    filipino = str(request.form['filipino'])
    french = str(request.form['french'])
    german = str(request.form['german'])
    hawaiian = str(request.form['hawaiian'])
    hungarian = str(request.form['hungarian'])
    indonesian = str(request.form['indonesian'])
    italian = str(request.form['italian'])
    japanese = str(request.form['japanese'])
    mexican = str(request.form['mexican'])
    russian = str(request.form['russian'])
    thai = str(request.form['thai'])

    day1 = str(request.form['1-day-or-more'])
    minutes15 = str(request.form['15-minutes-or-less'])
    minutes30 = str(request.form['30-minutes-or-less'])
    hours4 = str(request.form['4-hours-or-less'])
    minutes60 = str(request.form['60-minutes-or-less'])

    apples = str(request.form['apples'])
    avocado = str(request.form['avocado'])
    bananas = str(request.form['bananas'])
    citrus = str(request.form['citrus'])
    corn = str(request.form['corn'])
    grapes = str(request.form['grapes'])
    lemon = str(request.form['lemon'])
    mango = str(request.form['mango'])
    melons = str(request.form['melons'])
    peaches = str(request.form['peaches'])
    pears = str(request.form['pears'])
    pineapple = str(request.form['pineapple'])

    bacon = str(request.form['bacon'])
    beef = str(request.form['beef'])
    chicken = str(request.form['chicken'])
    clams = str(request.form['clams'])
    crab = str(request.form['crab'])
    duck = str(request.form['duck'])
    fish = str(request.form['fish'])
    pork = str(request.form['pork'])
    vegetables = str(request.form['vegetables'])
    cheese = str(request.form['cheese'])
    eggs = str(request.form['eggs'])
    breads = str(request.form['breads'])
    pasta = str(request.form['pasta'])
    spaghetti = str(request.form['spaghetti'])
    rice = str(request.form['rice'])
    potatoes = str(request.form['potatoes'])

    beginnercook = str(request.form['beginner-cook'])
    breakfast = str(request.form['breakfast'])
    lunch = str(request.form['lunch'])
    brunch = str(request.form['brunch'])
    beverages = str(request.form['beverages'])

    # untuk menambah data dari user input ke data original
    d = {'tester': [african, american, asian, australian, chinese, european, filipino, french, german, hawaiian, hungarian, indonesian, italian, japanese, mexican, russian, thai, apples, avocado, bananas, citrus, corn, grapes, lemon, mango, melons,
                    peaches, pears, pineapple, bacon, beef, chicken, clams, crab, duck, fish, pork, vegetables, cheese, eggs, breads, pasta, spaghetti, rice, potatoes, beginnercook, breakfast, lunch, brunch, beverages, day1, minutes15, minutes30, hours4, minutes60]}
    dftester = pd.DataFrame(data=d)
    dftester.to_csv('/Users/Ricky/Desktop/Jupyter/expo/testerpunya1234.csv')

    dftester1 = pd.read_csv(
        '/Users/Ricky/Desktop/Jupyter/expo/testerpunya1234.csv')
    dftester1 = dftester1[dftester1.tester != 'kosong']
    dftester1 = dftester1.drop(columns='Unnamed: 0')
    dftester1 = dftester1['tester'].str.cat(sep=',')
    dftester1 = dftest.loc[len(df.index)] = [
        'tester', dftester1, 9]

    # bikin data jadi dummies
    df_tags = pd.Series(dftest['tags']).str.get_dummies(',')
    # ambil name dan jumlah ingredients
    recipe = dftest[['name', 'n_ingredients']]

    # menggabungkan data frame yg nama dan jumlah bersama data dummies dari tags
    df_final = pd.concat([df_tags, recipe], axis=1)

    # mulai algoritma stuff
    X = df_final.iloc[:, :-2]
    y = df_final['n_ingredients']

    X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(
        X, y, test_size=0.2, random_state=1)

    knn = KNeighborsClassifier(n_neighbors=20)
    knn.fit(X_train_knn, y_train_knn)

    test = df_final.iloc[-1:, :-2]
    X_val = df_final.iloc[:-1, :-2]
    y_val = df_final['n_ingredients'].iloc[:-1]

    n_knn = knn.fit(X_val, y_val)

    distances, indeces = n_knn.kneighbors(test)

    final_table = pd.DataFrame(n_knn.kneighbors(test)[
                               0][0], columns=['distance'])
    final_table['index'] = n_knn.kneighbors(test)[1][0]
    result = final_table.join(df, on='index')
    html = result[['name', 'ingredients', 'steps', 'minutes']].to_html()

    # write html to file
    text_file = open(
        "templates/tester.html", "w")
    text_file.write(html)
    text_file.close()

    return render_template('tester.html')


@app.route('/generate', methods=['POST'])
def generate():

    foodname = int(request.form['foodname'])

    df_tags = pd.Series(dftest['tags']).str.get_dummies(',')
    recipe = dftest[['name', 'n_ingredients']]
    df_final = pd.concat([df_tags, recipe], axis=1)

    X = df_final.iloc[:, :-2]
    y = df_final['n_ingredients']
    X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(
        X, y, test_size=0.2, random_state=1)
    knn = KNeighborsClassifier(n_neighbors=20)
    knn.fit(X_train_knn, y_train_knn)
    test = df_final.iloc[foodname-1:foodname, :-2]
    X_val = df_final.iloc[:-1, :-2]
    y_val = df_final['n_ingredients'].iloc[:-1]

    n_knn = knn.fit(X_val, y_val)

    distances, indeces = n_knn.kneighbors(test)

    final_table = pd.DataFrame(n_knn.kneighbors(test)[
                               0][0], columns=['distance'])
    final_table['index'] = n_knn.kneighbors(test)[1][0]
    result = final_table.join(df, on='index')
    html = result[['name', 'ingredients', 'steps', 'minutes']].to_html()

    # write html to file
    text_file = open(
        "templates/tester.html", "w")
    text_file.write(html)
    text_file.close()

    return render_template('tester.html')


if __name__ == '__main__':
    app.debug = True
    app.run()
