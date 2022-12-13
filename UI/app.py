import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import flask
import difflib
from flask import Flask,request,render_template

ratings=pd.read_csv('/Users/devanshu/Movies_Recommendation/UI/Ratings.csv')
movies=pd.read_csv('/Users/devanshu/Movies_Recommendation/UI/movies.csv')
userdata=pd.read_csv('/Users/devanshu/Movies_Recommendation/UI/user_data.csv')
d=pd.merge(userdata,ratings,on='UserID',how='left')
df=pd.merge(d,movies,on='MovieID',how='left')


e = pd.read_csv('/Users/devanshu/Movies_Recommendation/UI/svd_p.csv')    ##Read predicted(USER BASED) dataset

#Recommend Top N to User
def recommend(userid_, n):
  f=e.loc[e[(e.uid==userid_)].est.sort_values(ascending=False).index][:n].iid.tolist()
  p=movies.loc[movies[movies.MovieID.isin(f)].index]
  k = ratings.loc[ratings[(ratings.UserID==userid_)].index]['MovieID'].tolist()
  return p.loc[p[~p.MovieID.isin(k)].index]

def rec(a,b):
  t = recommend(a,b)[['MovieID','Title','Genres']]
  t1,t2,t3 = t['MovieID'].to_list(),t['Title'].to_list(),t['Genres'].to_list()
  return t1,t2,t3

"""***Content Based (Genres)***"""

#Split Movies Genres in str
movies['Genres_'] = movies['Genres'].str.split('|')
movies['Genres_'] = movies['Genres'].fillna("").astype('str')
# movies.head(50)


tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0)
matrix = tf.fit_transform(movies['Genres_'])
csim = linear_kernel(matrix,matrix)

titles = movies['Title']
indices = pd.Series(movies.index, index=movies['Title'])

#Recommend Similar Movies
def c_rec(title):
    idx = indices[title]
    sim_sc = list(enumerate(csim[idx]))
    sim_sc = sorted(sim_sc, key=lambda x: x[1], reverse=True)[1:]
    movie_idx = [i[0] for i in sim_sc]
    print('Movies Similar to : ',title)
    yy=pd.DataFrame(movies.Title.iloc[movie_idx])[:10]
    return yy
# c_rec('Dracula: Dead and Loving It (1995)')

"""***CountVectorizer***"""

#fit and Trnasform the data into Vectorizer
cv = CountVectorizer()
cv_matrix = cv.fit_transform(movies['Genres'])
simc = cosine_similarity(cv_matrix, cv_matrix)

movies_ = movies.reset_index()
indices = pd.Series(movies_.index, index=movies_['Title'])
all_titles = [movies['Title'][i] for i in range(len(movies_['Title']))]

# Recommendation
def get_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(simc[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    tit = movies['Title'].iloc[movie_indices]
    dat = movies['Genres'].iloc[movie_indices]
    return_df = pd.DataFrame(columns=['Title','Genres'])
    return_df['Title'] = tit
    return_df['Genres'] = dat
    return return_df

app = flask.Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET', 'POST'])
def index(): return render_template('front.html')


@app.route("/predict/",methods=["POST"])
def main():
  if request.method == 'POST': m_name = request.form['movie_name'].title()
  if request.method == "GET" : m_name = request.args.get('movie_name').title()
  print(m_name)
  if m_name not in all_titles: return(render_template('front.html'))
  else:
    result_final = get_recommendations(m_name)
    names = []
    dates = []
    for i in range(len(result_final)):
      names.append(result_final.iloc[i][0])
      dates.append(result_final.iloc[i][1])
    return flask.render_template('back.html',result = 1,ret = zip(names,dates),search_name=m_name)


@app.route("/user/",methods=["POST"])
def userPred():
  print(request.method)

  try: a = int(request.form['user_id'])
  except: return render_template('front.html')
  print(a)
  b = 30
  t1,t2,t3 = rec(a,b)
  return render_template('back.html',neut = 1,ret = zip(t1,t2,t3))

if __name__ == '__main__':
    app.run(debug=True)