import  pickle
fr = open('vot-otb.pkl','rb')
inf = pickle.load(fr,encoding='latin-1')
fr.close()
print(inf['vot2014/ball'])