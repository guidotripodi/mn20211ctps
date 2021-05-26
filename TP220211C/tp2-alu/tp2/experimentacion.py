import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, recall_score, precision_score
from tqdm import tqdm
import numpy as np
from tqdm import tqdm


data = pd.DataFrame(pd.read_csv('train.csv'))
data = data.to_numpy()


# me quedo con el 85% de los datos para train
np.random.shuffle(data)
limit = int(0.85*data.shape[0])

X_train = data[0:limit, 1:]
y_train = data[0:limit, 0]

X_test = data[limit:, 1:]
y_test = data[limit:, 0]

accuracies = []
for k in tqdm(np.arange(1,65,2)):
    knn = KNeighborsClassifier(k)
    knn.fit(X_train,y_train)
    pred = knn.predict(X_test)
    accuracies.append(accuracy_score(y_test,pred))

plt.plot(np.arange(1,65,2), accuracies)
plt.grid()
plt.xlabel('valor de k')
plt.ylabel('Accuracy %')
plt.title('Accuracy vs k')
plt.savefig('Accuracy para valores de k.png')



#veamos otras metricas para los mejores resultados
valores = np.arange(1,10)
rec = []
prec = []
accr = []

for k in tqdm(valores):
    knn = KNeighborsClassifier(k)
    knn.fit(X_train,y_train)
    pred = knn.predict(X_test)
    rec.append(recall_score(y_test,pred,average=None).mean())
    prec.append(precision_score(y_test,pred,average=None).mean())
    accr.append(accuracy_score(y_test,pred))

plt.plot(valores,accr,'b')
plt.plot(valores,prec,'r')
plt.plot(valores,rec,'g')
plt.legend(['Accuracy','Precision','Recall'])
plt.grid()
plt.xlabel('Valor de k')
plt.ylabel('Scores')
plt.savefig('Metricas para valores de k.png')

""" usando K-fold-cross-validation con 3 vecinos """
X = data[:, 1:]
y = data[:, 0]
knn = KNeighborsClassifier(n_neighbors=3)

scores = []
for n_splits in range(2, 10):
    score = cross_val_score(knn, X, y, cv=n_splits, scoring='accuracy')
    scores.append(score.mean())

rango = np.arange(2, 10)
plt.plot(rango, scores, 'b')
plt.xlabel('Valor de K')
plt.ylabel('Promedio de accuracy')
plt.title('K-Fold cross validation para 3 vecinos')
plt.grid()
plt.savefig('K-fold para 3 vecinos.png')

# veamos la distribucion de los elementos:
elementos = []
for i in range(0, 10):
    elementos.append(np.count_nonzero(y == i))

plt.bar(np.arange(0, 10), elementos)
plt.xticks(list(range(0, 10)))
plt.xlabel('Categoria')
plt.ylabel('Cantidad de elementos')
plt.title('Cantidad de elementos por categoria')
plt.savefig('Distribucion de categorias.png')

""" resultados en el formato de la competencia de kaggle
res = pd.Series(y_pred, name="Label")
submission = pd.concat([pd.Series(range(1, 28001), name="ImageID"), res], axis=1)
submission.to_csv("submission.csv",index=False)

"""