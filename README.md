# movie_recommendation_system
A recommendation system is a type of information filtering system that makes suggestions based on user preferences and makes an effort to forecast the preferences of the user.

There are several algorithms and models in order to implement different recommendation system approaches such as Collaborative Filtering and Content based filtering.
In item based collaborative filtering, implement model based algorithms that are KNNWithMeans, KNNBasic and KNNBaseline. Checking their performance using RMSE and MAE from this came to the conclusion that RMSE and MAE of KNNWithMeans is much better than other algorithms because of the lowest RMSE value.
In user based collaborative filtering, implemented the above same algorithms along with matrix factorization based algorithm SVD (Singular Value Decomposition). Checked their RMSE and
MAE values. From this we came to the conclusion that performance of SVD is better than other algorithms.

In the above project approach, movie genres have been taken into account; however, in the future, also take user age into account because movie preferences change with age. For instance, when we are young, we tend to prefer animated films over other types of films. The memory requirements of the suggested approach need to be improved in the future.
Here, the suggested methodology has only been applied to several movie datasets. The performance can be computed in the future, and it can also be used to the Film Affinity and Netflix databases.
