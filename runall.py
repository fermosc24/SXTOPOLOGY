#!/usr/bin/env python
# coding: utf-8

# In[1]:


from mygraph import *
import pyconll
import glob
from random import sample,seed,shuffle
from collections import defaultdict
import pandas as pd
import re
from dfply import *
import seaborn as sns
import matplotlib.pyplot as plt
import plotnine as p9


# ## 1. Set corpora files
# 
# Set `PATHA` to the directory where `Universal Dependencies 2.11` is stored.

# In[2]:


PATHA = '/Users/ferminmoscosodelpradomartin/CORPORA/'
PATHB = "Universal Dependencies 2.11/ud-treebanks-v2.11/"
PATHC = "*/*.conllu"
PATH=PATHA+PATHB+PATHC

files = sorted(glob.glob(PATH))

langFiles = defaultdict(list)
langNames = defaultdict(set)

for f in files:
    name = f.removeprefix(PATHA+PATHB)
    langcode = name.split("/")[-1].split("-")[0].split("_")[0]
    langname = name.split("/")[0].removeprefix("UD_").split("-")[0]
    langFiles[langcode].append(f)
    langNames[langcode].add(langname)


# ## 2. Precompute extrema
# 
# Compute values of $\max_N h_\mathrm{deg}$, $\min_N h_\mathrm{deg}$, $\max_N h_\mathrm{ks}$, and $\min_N h_\mathrm{ks}$ for $N=4 \ldots 50$. 

# In[3]:


maxL = 50
MinMax={}
for i in range(4,maxL+1):
    if not i in MinMax:
        mHdeg = MinHdeg(i)
        MHdeg = MaxHdeg(i)
        mHks  = MinHks(i)
        MHks  = MaxHks(i)
        MinMax[i] = (mHks,MHks,mHdeg,MHdeg)


# ## 3. Main computations
# 
# From each corpus file:
# - Sample `N2` sentences from all corpora for which there are at least `N1` valid sentences (trees with between 4 and 50 nodes).
# - For each selected tree:
#     - Compute their $h_\mathrm{deg}$, and $h_\mathrm{ks}$ values
#     - Estimate the value of the preferential attachment exponent $\hat\alpha$
#     - Generate a uniformly sampled random tree matched in number of nodes, and compute its $h_\mathrm{deg}$, $h_\mathrm{ks}$, and $\hat\alpha$ values
# - For each of the languages:
#     - Compute the mean value $\langle \hat\alpha \rangle$ for the language
#     - For each of the trees selected from that language:
#         - Generate a random tree matched in number of nodes using nonlinear preferential attachment with exponent $\langle \hat\alpha \rangle$,  and compute its $h_\mathrm{deg}$, and $h_\mathrm{ks}$ values
#     - Select `N3` of the random trees selected above and perform `Nepochs` of optimization on those, with parameter $\rho=$ `Rho` and $\sigma$=`NoiseL`.
#         - Store the intermediate optimization results in `trajectories`
#     

# In[4]:


N1=50   # Threshold for corpus size (i.e, min number of sentences)
N2=1000 # Max sample size from each corpus to estimate entropies 
        # (N2 >= N1)
N3=100  # Max sample size for optimization
Nepochs = 400 # Number of epochs for optimization
NoiseL = .075 # Noise level for optimization
Rho = .9
cond2 = "optimized (noise=.%2d,rho=.%2d)"%(100*NoiseL,100*Rho)

result = pd.DataFrame({"Lang":[],
                       "LCode":[],
                       "Type":[],
                       "H_ks":[],
                       "H_deg":[],
                       "Length":[],
                       "mHks":[],"MHks":[],"mHdeg":[],"MHdeg":[],
                       "Alpha":[]})
trajectories = pd.DataFrame({"Lang":[],"Epoch":[],
                             "H_ks":[],"H_deg":[],"KLDiv":[]})

for language in langFiles:
    corpusf = []
    langName = "/".join(list(langNames[language]))
    print("LANGUAGE:",langNames[language])
    for f in langFiles[language]:
        name = f.removeprefix(PATHA+PATHB)
        cit = pyconll.load_from_file(f)
        corpusf += [s for s in cit]
    shuffle(corpusf)
    if len(corpusf)>=N1:
        seed(123)
        nsent = 0
        corpusf2 = []
        for s in corpusf:
            t = CONLL2DG(s)
            if len(t.nodes)>3 and                len(t.nodes)==max(t.nodes)+1 and                len(t.nodes)<=maxL:
                corpusf2.append(t)
                nsent+=1
            if nsent>=N2:
                break
        corpusf = corpusf2
        if nsent>=N1:
            print("GOOD",langName,nsent)
            rtrees = []
            lens = []
            alphas = []
            for tree in corpusf:
                h = EntropyKS(tree)
                s = EntropyD(DegreeDist(tree,50,directed=True))
                l = len(tree.nodes)
                minmax = MinMax[l]
                lens.append((l,minmax))
                
                maxdegree = np.max(list(zip(*tree.degree(list(range(l)))))[1])
                Alpha = 1-np.log(np.log(l))/np.log(maxdegree)
                alphas.append(Alpha)
                
                result.loc[len(result)] =                     (langName,language,"real",h,s,l)+                                        minmax+(Alpha,)
                tree = RandomDiTree(l)

                maxdegree = np.max(list(zip(*tree.degree(list(range(l)))))[1])
                AlphaR = 1-np.log(np.log(l))/np.log(maxdegree)

                rtrees.append((tree,l,minmax))
                hr = EntropyKS(tree)
                sr = EntropyD(DegreeDist(tree,50,directed=True))
                result.loc[len(result)] =                     (langName,language,"baseline",hr,sr,l)+                    minmax+(AlphaR,)
            Alpha = np.mean(alphas)
            print(Alpha)
            for l,minmax in lens:
                tree = BiasedTree(l,Alpha)

                maxdegree = np.max(list(zip(*tree.degree(list(range(l)))))[1])
                AlphaR = 1-np.log(np.log(l))/np.log(maxdegree)

                hr = EntropyKS(tree)
                sr = EntropyD(DegreeDist(tree,50,directed=True))
                result.loc[len(result)] =                     (langName,language,"attachment",hr,sr,l)+                    minmax+(AlphaR,)
            refdist = result.loc[(result["Lang"]==langName)&                           (result["Type"]=="real"),["H_ks","H_deg"]]
            refdist = np.array(refdist)[:N3,:]
            # Here optimize rtrees
            rtrees,Ls,Minmax = list(zip(*(rtrees[:N3])))
            _,HS,_,derror = OptimizeL(rtrees,
                                      nepochs=Nepochs,
                                      rho=Rho,
                                      npred=200,noise=NoiseL,
                                      Href=refdist)
            for i,(m1,M1,m2,M2) in enumerate(Minmax):
                hks,hdeg = tuple(HS[i,:,-1])
                tup = (langName,language,cond2)+                  (hks,hdeg)+                   (Ls[i],m1,M1,m2,M2,AlphaR,)
                result.loc[len(result)] = tup
            steps = np.mean(HS,axis=0)
            trajN = pd.DataFrame({"Lang":[langName]*(Nepochs+1),
                             "Epoch":np.arange(Nepochs+1),
                             "H_ks":steps[0,:],
                             "H_deg":steps[1,:],
                             "KLDiv":derror})
            trajectories = pd.concat([trajectories,trajN])

result.to_csv("langdata.csv")
trajectories.to_csv("trajdata.csv)


# ## 4. Paired $t$-tests

# In[5]:


from scipy.stats import ttest_rel
print("Ttest Hdeg:",ttest_rel(result.loc[result["Type"]=="real",["H_deg"]],
                result.loc[result["Type"]=="baseline",["H_deg"]]))
print("Ttest Hks:",ttest_rel(result.loc[result["Type"]=="real",["H_ks"]],
                result.loc[result["Type"]=="baseline",["H_ks"]]))

print(result.loc[result["Type"]=="baseline",
                 ["H_deg","H_ks"]] >> \
    summarize_each([np.mean,
                    lambda x:np.std(x)/np.sqrt(len(x))],
                   X.H_deg,X.H_ks))
print(result.loc[result["Type"]=="real",
                 ["H_deg","H_ks"]] >> \
    summarize_each([np.mean,
                    lambda x:np.std(x)/np.sqrt(len(x))],
                   X.H_deg,X.H_ks))


# ## 5. Generate Figure S2
# 
# - For each of the uniform random trees, sample a nonlinear preferential attachment tree, matched in number of nodes, using the $\langle \hat\alpha \rangle$ from the relevant language.
# - Plot the relevant distribution

# In[6]:


baselines = result.loc[result["Type"]=="baseline"]
baselines2 = baselines.copy()
baselines2["Type"]="regenerated"
baselines2.head()
AlphaR = []
hks = []
hdeg = []
for i in range(baselines2.shape[0]):
    l = baselines2.iloc[i]["Length"]
    alpha = baselines2.iloc[i]["Alpha"]
    tree = BiasedTree(l,alpha)
    maxdegree = np.max(list(zip(*tree.degree(list(range(l)))))[1])
    AlphaR.append(1-np.log(np.log(l))/np.log(maxdegree))
    hks.append(EntropyKS(tree))
    hdeg.append(EntropyD(DegreeDist(tree,50,directed=True)))
baselines2["H_ks"] = hks
baselines2["H_deg"] = hdeg
baselines2["AlphaR"] = AlphaR
baselines["AlphaR"] = 0 
baselines2 = pd.concat([baselines,baselines2])


# In[20]:


colors = {'baseline': sns.color_palette("colorblind")[4],
          'regenerated': sns.color_palette("colorblind")[1]}

baselines2 = baselines2 >> mutate(H_ks=(X.H_ks-X.mHks)/(X.MHks-X.mHks),
                                  H_deg=(X.H_deg-X.mHdeg)/(X.MHdeg-X.mHdeg))

g=sns.jointplot(data=baselines2,
                kind="scatter",marker=".",
                y="H_ks",x="H_deg",hue="Type",alpha=.2,
                marginal_kws = dict(common_norm=False),
                palette=colors)
g.plot_joint(sns.kdeplot, zorder=0, levels=6)
plt.ylabel(r"$H_\mathrm{ks}$")
plt.xlabel(r"$H_\mathrm{deg}$")
plt.grid()
plt.savefig('FigS2.png',dpi=300)
plt.savefig('FigS2.pdf', dpi=300)
plt.show()


# ## 6. Kullback-Leibler Divergences
# 
# Here, I plot the evolution of the Kullback-Leibler divergence (KLD) between the real dependency graphs and the gradually optimized ones. In addition, I compute and add to the plots the following:
# - KLDs between the real dependency graphs and those generated by sublinear preferential attachment.
# - KLDs between identical distributions of real dependency graphs, to serve as a comparison baseline (note that the KLDs are estimated using a parametric model *assuming* that the $H_\mathrm{ks}$ and $H_\mathrm{deg}$ values follow a bivariate Gaussian, which is only an approximation).

# In[10]:


kldt = trajectories >>         group_by(X.Epoch) >>         summarize(
            KLDiv=X.KLDiv.mean(),
            H_deg=X.H_deg.mean(),
            H_ks=X.H_ks.mean(),
            SEKL=X.KLDiv.std(),
            SEdeg=X.H_deg.std(),
            SEks=X.H_ks.std(),
            N=n(X.KLDiv)) >> \
            mutate(SEKL=X.SEKL/(X.N**.5),
                   SEdeg=X.SEdeg/(X.N**.5),
                   SEks=X.SEks/(X.N**.5)) >> \
            mutate(UEBKL = X.KLDiv + 1.96*X.SEKL,
                   LEBKL = X.KLDiv - 1.96*X.SEKL,
                   UEBdeg = X.H_deg + 1.96*X.SEdeg,
                   LEBdeg = X.H_deg - 1.96*X.SEdeg,
                   UEBks = X.H_ks + 1.96*X.SEks,
                   LEBks = X.H_ks - 1.96*X.SEks)


# In[15]:


def AuxF(df):
    m1 = np.array(df.loc[df["Type"]=="real",["H_ks","H_deg"]])
    m2 = np.array(df.loc[df["Type"]=="attachment",["H_ks","H_deg"]])
    return MyKLD(m1,m2)

def AuxF2(df,maxN=None):
    kld = []
    langs = set(df["Lang"])
    nsim = len(langs)
    for lang in langs:
        df2 = df.loc[df["Lang"]==lang]
        ssize = len(df2)
        m1 = np.array(df2[["H_ks","H_deg"]])
        m2 = np.array(df2[["H_ks","H_deg"]].sample(ssize,replace=True))
        if maxN is None:
            kld.append(MyKLD(m1,m2))
        else:
            kld.append(MyKLD(m1[:maxN,:],m2[:maxN,:]))
    return (np.mean(kld),1.96*np.std(kld)/np.sqrt(nsim))

kldsl = result.loc[result["Type"].isin(["real","attachment"])] >>         group_by(X.Type,X.Lang) >> head(100) >> ungroup()

klds = []
for l in set(kldsl["Lang"]):
    subset = kldsl.loc[kldsl["Lang"]==l]
    klds.append(AuxF(subset))

kldattach = np.mean(klds)
kldattachs = 1.96*np.std(klds)/np.sqrt(len(klds))
ls = list((result.loc[result["Type"]=="real"] >> group_by(X.Lang) >> summarize(N=X.Type.count()))["N"])
kldBSm,kldBSs = AuxF2(result.loc[result["Type"]=="baseline"],100)


# Generate Fig. 3B:

# In[16]:


plt.plot(kldt["Epoch"],kldt["KLDiv"],"-r",label="optimization")
plt.fill_between(kldt["Epoch"], kldt["LEBKL"], kldt["UEBKL"],
                 where= kldt["UEBKL"] >= kldt["LEBKL"],
                 facecolor='red', alpha=.4,
                 interpolate=True)
plt.ylim((0,.28))

a = np.array([kldattach-kldattachs]*len(kldt["Epoch"]))
b = np.array([kldattach+kldattachs]*len(kldt["Epoch"]))
c = np.array([kldattach]*len(kldt["Epoch"]))
plt.fill_between(kldt["Epoch"], a, b,
                 where= b >= a,
                 facecolor='blue', alpha=.4,
                 interpolate=True)
plt.plot(kldt["Epoch"],c,"b-",
             label="sublinear preferential attachment")

a = np.array([kldBSm-kldBSs]*len(kldt["Epoch"]))
b = np.array([kldBSm+kldBSs]*len(kldt["Epoch"]))
c = np.array([kldBSm]*len(kldt["Epoch"]))
plt.fill_between(kldt["Epoch"], a, b,
                 where= b >= a,
                 facecolor='grey', alpha=.4,
                 interpolate=True)
plt.plot(kldt["Epoch"],c,color="grey",label="identical distributions")

plt.grid()
plt.ylabel("Kullback-Leibler Divergence")
plt.xlabel("Epoch")
plt.legend(loc="upper right")

plt.savefig('Fig3B.png',dpi=300)
plt.savefig('Fig3B.pdf', dpi=300)
plt.show()


# Recompute baseline KLDs without the 100 sentence limit:

# In[17]:


kldsl = result.loc[result["Type"].isin(["real","attachment"])] >>         group_by(X.Type,X.Lang) >> ungroup()

klds = []
for l in set(kldsl["Lang"]):
    subset = kldsl.loc[kldsl["Lang"]==l]
    klds.append(AuxF(subset))

kldattach = np.mean(klds)
kldattachs = 1.96*np.std(klds)/np.sqrt(len(klds))
ls = list((result.loc[result["Type"]=="real"] >> group_by(X.Lang) >> summarize(N=X.Type.count()))["N"])
kldBSm,kldBSs = AuxF2(result.loc[result["Type"]=="baseline"])


# Compute distributions for fixed $\alpha$ values between 0.0 and 2.0:

# In[14]:


res = result.loc[result["Type"]=="real",["Lang","Length","H_ks","H_deg"]]
alphaTest = pd.DataFrame({"Lang":[],"Alpha":[],"KLD":[]})
Alphas = np.linspace(0,2,21)
Langs = set(res["Lang"])
for lang in Langs:
    print(lang)
    res2 = res.loc[res["Lang"]==lang,["Length","H_ks","H_deg"]]
    dim0 = len(res2)
    m1 = np.array(res2[["H_ks","H_deg"]])
    for alpha in Alphas:
        hks2=[]
        hdeg2=[]
        for i in range(dim0):
            l = res2["Length"].iloc[i]
            tree = BiasedTree(l,alpha)
            hks = EntropyKS(tree)
            hdeg = EntropyD(DegreeDist(tree,50,directed=True))
            hks2.append(hks)
            hdeg2.append(hdeg)
        m2 = np.transpose(np.array([hks2,hdeg2]))
        alphaTest.loc[len(alphaTest)] =(lang,alpha,MyKLD(m1,m2))


# Generate Fig. S3

# In[19]:


alphaMeans = alphaTest[["Alpha","KLD"]] >> group_by(X.Alpha) >>              summarize_each([np.mean,np.std],X.KLD)
alphaMeans["KLD_std"] /= 124**.5
alphaMeans["eKLD"] = 1.96*alphaMeans["KLD_std"]

a = np.array([kldattach-kldattachs]*len(Alphas))
b = np.array([kldattach+kldattachs]*len(Alphas))
c = np.array([kldattach]*len(Alphas))

a2 = np.array(alphaMeans["KLD_mean"]-alphaMeans["eKLD"])
b2 = np.array(alphaMeans["KLD_mean"]+alphaMeans["eKLD"])


plt.plot(alphaMeans["Alpha"],alphaMeans["KLD_mean"],"g-",
         label=r"Fixed $\alpha$")
plt.fill_between(Alphas, a2, b2,
                 where= b2 >= a2,
                 facecolor='green', alpha=.4,
                 interpolate=True)
plt.plot(Alphas,c,"b-",label=r"Optimal $\alpha$")
plt.fill_between(Alphas, a, b,
                 where= b >= a,
                 facecolor='blue', alpha=.4,
                 interpolate=True)

a3 = np.array([kldBSm-kldBSs]*len(Alphas))
b3 = np.array([kldBSm+kldBSs]*len(Alphas))
c3 = np.array([kldBSm]*len(Alphas))
plt.fill_between(Alphas, a3, b3,
                 where= b3 >= a3,
                 facecolor='grey', alpha=.4,
                 interpolate=True)
plt.plot(Alphas,c3,color="grey",label="Identical Distributions")

plt.grid()
plt.ylabel("Kullback-Leibler Divergence")
plt.xlabel(r"$\alpha$")
plt.legend(loc="upper left")

plt.vlines(1.0,-0.1,1,linestyles="dashed",color="black")

plt.ylim(0,.61)

plt.savefig('FigS3.png',dpi=300)
plt.savefig('FigS3.pdf', dpi=300)
plt.show()


# ## 7. Convergence Analysis
# 
# Plot Fig. 3A, showing the convergence of both entropy measures in the optimization algorithm:

# In[21]:


plt.plot(kldt["Epoch"],kldt["H_deg"],"-",label=r"$h_\mathrm{deg}$",
        color="cyan")
plt.fill_between(kldt["Epoch"], kldt["LEBdeg"], kldt["UEBdeg"],
                 where= kldt["UEBdeg"] >= kldt["LEBdeg"],
                 facecolor='cyan', alpha=.4,
                 interpolate=True)

plt.plot(kldt["Epoch"],kldt["H_ks"],"-",label=r"$h_\mathrm{ks}$",
         color="pink")
plt.fill_between(kldt["Epoch"], kldt["LEBks"], kldt["UEBks"],
                 where= kldt["UEBks"] >= kldt["LEBks"],
                 facecolor='pink', alpha=.4,
                 interpolate=True)
plt.legend(loc="center right")

#plt.ylim((0,.28))
plt.grid()
plt.ylabel("Entropy (bits)")
plt.xlabel("Epoch")
plt.savefig('Fig3A.png',dpi=300)
plt.savefig('Fig3A.pdf', dpi=300)
plt.show()


# Compute average results by language and type:

# In[22]:


average = result >>         mutate(H_ks = (X.H_ks-X.mHks)/(X.MHks-X.mHks),               H_deg = (X.H_deg-X.mHdeg)/(X.MHdeg-X.mHdeg)) >>        group_by(X.Lang,X.Type) >>        summarize(Length = np.mean(X.Length),                  H_ks = np.mean(X.H_ks),                  H_deg = np.mean(X.H_deg),
                  Alpha = np.mean(X.Alpha),
                  NSents = n(X.Alpha),
                  SE_H_ks = np.std(X.H_ks),
                  SE_H_deg = np.std(X.H_deg),
                  SE_Alpha = np.std(X.Alpha)) >>\
        mutate(SE_H_ks = X.SE_H_ks*(X.NSents**-.5),
               SE_H_deg = X.SE_H_deg*(X.NSents**-.5),
               SE_Alpha = X.SE_Alpha*(X.NSents**-.5))


# Plot Fig. 2:

# In[23]:


average.loc[average["Type"]==cond2,"Type"]="optimized"
average["Type"] = average["Type"].astype('category')
average["Type"].cat.reorder_categories(['real', 'attachment', 'baseline', 'optimized'])

colors = {'real': sns.color_palette("colorblind")[0], 
          'attachment': sns.color_palette("colorblind")[8], 
          'baseline': sns.color_palette("colorblind")[4],
          'optimized': sns.color_palette("colorblind")[3]}


sns.scatterplot(data=average,alpha=.7,#marker=".",
              y="H_ks",x="H_deg",hue="Type",palette=colors)
sns.set_palette("colorblind")
plt.xlim((0,1))
plt.ylabel(r"$\langle H_\mathrm{ks} \rangle$")
plt.ylim((0,1))
plt.xlabel(r"$\langle H_\mathrm{deg} \rangle$")
plt.grid()
plt.savefig('Fig2.png',dpi=300)
plt.savefig('Fig2.pdf', dpi=300)
plt.show()


# ## 8. Logistic Classifiers

# In[26]:


result2 = result >>         mutate(H_ks = (X.H_ks-X.mHks)/(X.MHks-X.mHks),               H_deg = (X.H_deg-X.mHdeg)/(X.MHdeg-X.mHdeg))


# In[36]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

log_data = np.array(result2.loc[(result2["Length"])&                                (result2["Type"].isin(["real",
                                                       "baseline"])),
                       ["H_ks","H_deg"]])
log_cat = np.array(result2.loc[(result2["Length"])&                               (result2["Type"].isin(["real",
                                                      "baseline"])),
                        ["Type"]])=="real"

x_train, x_test, y_train, y_test =     train_test_split(log_data, log_cat, test_size=0.1, random_state=13)
logisticRegr = LogisticRegression()
logisticRegr.fit(x_train, y_train)
score = logisticRegr.score(x_test, y_test)
print("LR distinguishing real from baseline graphs: %d%%"%      (int(np.round(100*score,0)),))


# In[42]:


log_data = np.array(result2.loc[(result2["Length"]>=10)&                                (result2["Type"].isin(["real",
                                                       "baseline"])),
                       ["H_ks","H_deg"]])
log_cat = np.array(result2.loc[(result2["Length"]>=10)&                               (result2["Type"].isin(["real",
                                                      "baseline"])),
                        ["Type"]])=="real"

x_train, x_test, y_train, y_test =     train_test_split(log_data, log_cat, test_size=0.1, random_state=13)
logisticRegr = LogisticRegression()
logisticRegr.fit(x_train, y_train)
score = logisticRegr.score(x_test, y_test)
print("LR distinguishing real from baseline graphs: %d%% (for L>9)"%      (int(np.round(100*score,0)),))


# In[38]:


log_data = np.array(result2.loc[(result2["Length"])&                                (result2["Type"].isin(["real","attachment"])),
                       ["H_ks","H_deg"]])
log_cat = np.array(result2.loc[(result2["Length"])&                               (result2["Type"].isin(["real","attachment"])),
                        ["Type"]])=="real"

x_train, x_test, y_train, y_test =     train_test_split(log_data, log_cat, test_size=0.1, random_state=13)
logisticRegr = LogisticRegression()
logisticRegr.fit(x_train, y_train)
score = logisticRegr.score(x_test, y_test)
print("LR distinguishing real from sublinear pref. attachment graphs: %d%%"%      (int(np.round(100*score,0)),))


# In[43]:


log_data = np.array(result2.loc[(result2["Length"]>=10)&                                (result2["Type"].isin(["real","attachment"])),
                       ["H_ks","H_deg"]])
log_cat = np.array(result2.loc[(result2["Length"]>=10)&                               (result2["Type"].isin(["real","attachment"])),
                        ["Type"]])=="real"

x_train, x_test, y_train, y_test =     train_test_split(log_data, log_cat, test_size=0.1, random_state=13)
logisticRegr = LogisticRegression()
logisticRegr.fit(x_train, y_train)
score = logisticRegr.score(x_test, y_test)
print("LR distinguishing real from sublinear pref. attachment graphs: %d%% (for L>9)"%      (int(np.round(100*score,0)),))


# ## 9. Plot Fig. 4

# In[40]:


result3 = result2.loc[result2["Type"]==cond2] 
nr = result3.shape[0]
result3 = pd.concat([result3,
                     result2.loc[result2["Type"]=="real"].sample(nr),
                     result2.loc[result2["Type"]=="baseline"].sample(nr)])


# In[41]:


result2.loc[result2["Type"]==cond2,"Type"]="optimized"
resaux = result2.loc[result2["Type"]!="optimized"]
resaux["Type"] = resaux["Type"].astype('category')
resaux["Type"].cat.reorder_categories(['real', 'attachment','baseline'])

colors = {'real': sns.color_palette("colorblind")[0], 
          'attachment': sns.color_palette("colorblind")[8], 
          'baseline': sns.color_palette("colorblind")[4],
          'optimized': sns.color_palette("colorblind")[3]}


g=sns.jointplot(data=resaux,#.sample(10000),
                kind="scatter",marker=".",
                y="H_ks",x="H_deg",hue="Type",alpha=.2,
                marginal_kws = dict(common_norm=False),
                palette=colors)
g.plot_joint(sns.kdeplot, zorder=0, levels=6)
plt.ylabel(r"$H_\mathrm{ks}$")
plt.xlabel(r"$H_\mathrm{deg}$")
plt.grid()
plt.savefig('Fig4.png',dpi=300)
plt.savefig('Fig4.pdf', dpi=300)
plt.show()


# 10. Convergence analysis (II)
# 
# Plot Fig. S1

# In[50]:


trajs=trajectories>>group_by("Epoch")>>    summarize_each([np.mean],X.H_deg,X.H_ks)


# In[ ]:


Rho = np.round(np.linspace(0,1,11),2)
ntrees = 100
nepochs = 2000
lens = np.array(result.loc[(result["Type"]=="real")&                           (result["Lang"]=="English"),                           "Length"])[:ntrees]
convergence = pd.DataFrame({"rho":[],"Epoch":[],
                            "H_ks":[],"se_H_ks":[],
                            "H_deg":[],"se_H_deg":[]})
trees = [RandomDiTree(n) for n in lens]
mmax = np.array([MinMax[n] for n in lens])
for i,rho in enumerate(list(Rho)):
    print(rho)
    _,HS,_,_ = OptimizeL(trees,
                         nepochs=nepochs,
                         rho=rho,
                         npred=50,noise=0,
                         Href=None)
    HS[:,0,:] = np.transpose((np.transpose(HS[:,0,:])-mmax[:,0])                             /(mmax[:,1]-mmax[:,0]))
    HS[:,1,:] = np.transpose((np.transpose(HS[:,1,:])-mmax[:,2])                             /(mmax[:,3]-mmax[:,2]))
    means = np.mean(HS,axis=0)
    ses = np.std(HS,axis=0)/np.sqrt(ntrees)
    convN = pd.DataFrame({"rho":[rho]*(nepochs+1),
                          "Epoch":np.arange(nepochs+1),
                          "H_ks":means[0,:],
                          "se_H_ks":ses[0,:],
                          "H_deg":means[1,:],
                          "se_H_deg":ses[1,:]})
    convergence = pd.concat([convergence,convN])


# In[ ]:


_,HS,_,_ = OptimizeL(trees,
                     nepochs=400,
                     rho=.9,
                     npred=50,noise=.075,
                     Href=None)
HS[:,0,:] = np.transpose((np.transpose(HS[:,0,:])-mmax[:,0])                         /(mmax[:,1]-mmax[:,0]))
HS[:,1,:] = np.transpose((np.transpose(HS[:,1,:])-mmax[:,2])                         /(mmax[:,3]-mmax[:,2]))
means = np.mean(HS,axis=0)
ses = np.std(HS,axis=0)/np.sqrt(ntrees)
convG = pd.DataFrame({"Epoch":np.arange(401),
                      "H_ks":means[0,:],
                      "se_H_ks":ses[0,:],
                      "H_deg":means[1,:],
                      "se_H_deg":ses[1,:]})


locat = tuple(np.array(
    average.loc[(average["Lang"]=="English")&\
                (average["Type"]=="real")].iloc[0])[3:5])
locat=pd.DataFrame({"x":[locat[0]],"y":[locat[1]]})
locat



arrows = (convergence >> group_by(X.rho) >>          mutate(H_ks2=lead(X.H_ks),
                 H_deg2=lead(X.H_deg)) >>\
          ungroup()).dropna()

goodarrow = (convG >>             mutate(H_ks2=lead(X.H_ks),
                    H_deg2=lead(X.H_deg))).dropna()

plot=(p9.ggplot()+ p9.geom_segment(data = arrows, 
               mapping = p9.aes(x = "H_deg", y = "H_ks", 
                                xend = "H_deg2", yend = "H_ks2",
                                colour = "rho"),
               arrow = p9.arrow(angle = 15, type = "closed", 
                                length=.05))+
 p9.labs(colour = r"$\rho$")+
 p9.geom_segment(data = goodarrow, 
               mapping = p9.aes(x = "H_deg", y = "H_ks", 
                                xend = "H_deg2", yend = "H_ks2"),
               arrow = p9.arrow(angle = 15, type = "closed", 
                                length=.05),
               color="green")+
 p9.geom_point(data=locat,mapping=p9.aes(x="y",y="x"),
               shape="*",size=4)+
 p9.xlim(0,1)+p9.ylim(0,1)+p9.theme_bw()+
 p9.xlab(r"$\langle H_{\mathrm{deg}} \rangle$")+
 p9.ylab(r"$\langle H_{\mathrm{ks}} \rangle$")+
 p9.scale_color_gradient2(low='#2b83ba', mid='#ffffbf',
                          high='#d7191c',midpoint=.5))


plot.save(filename = 'FigS1.png', height=6, width=6, units = 'in', dpi=300)
plot.save(filename = 'FigS1.pdf', height=6, width=6, units = 'in', dpi=300)

