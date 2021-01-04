import os
import re
import string
import collections as c
import numpy as np

# input path and return a file list 
def create_file_list(path):
    f = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        f.extend(filenames)
        break
    return f

def create_corpus(path):
    f = create_file_list(path)
    corpus = {}
    for file in f:
        full_file = path+"/"+file
        # Using readlines() 
        afile = open(full_file, 'r',encoding="UTF-8-sig") 
        lines = afile.readlines() 
        for n,aline in enumerate(lines): 
            aline = aline.strip()
            if (n==0):
                header = aline.split("\t")
            else:
                content = aline.split("\t")
                # zip is a handy way to rapidly populate the dictionary
                record = dict(zip(header,content))
                article_id = record["UT"]
                corpus[article_id]=record
    return corpus

def create_metadata(path):
    metadata = {}
    f = create_file_list(path)

    for file in f:
        full_file = path+"/"+file

        afile = open(full_file, 'r',encoding="UTF-8-sig") 
        lines = afile.readlines() 
    
        for n,aline in enumerate(lines): 
            aline = aline.strip()
            if (n==0):
                header = aline.split("\t")
            else:
                content = aline.split("\t")
                # zip is a handy way to rapidly populate the dictionary
                record = dict(zip(header,content))
                article_id = record["UT"]
            
                cited = record["TC"]
                if (cited == ''):
                    cited = 0
                
                year = record["PY"]
                if (year == ''):
                    year = 2020

                metadata[article_id] = [int(year),int(cited)]
    return metadata

# Here we do it for just one field, the citations
# Maybe we need a config file
# maybe separate sockets for each field
# citations are annoying because they store ''
def annotate(lbl,corpus):
    # citation socket
    if (lbl == "F1"):
        index = citation_socket(corpus)
        
    # authorship and address socket
    if (lbl == "F2"):
        index = organisation_socket(corpus)

    # keyword, title and abstract socket
    if (lbl == "F3"):
        index = content_socket(corpus)
        
    return index

def citation_socket(corpus):
    index = corpus.copy()
    for art_id in index:
        content = corpus[art_id]
        cite_str = content["CR"]
        cite_list = cite_str.split("; ")
        content["F1"]=cite_list
        index[art_id]=content
    return index
        
def content_socket(corpus):
    index = corpus.copy()
    for art_id in index:
        content = corpus[art_id]
        title = content["TI"]
        database = content["ID"]
        keyword = content["DE"]
        abstract = content["AB"]
        abstract = remove_copyright(abstract)
  
        list1 = delist(" ",title)
        list2 = delist("; ",database)
        list3 = delist("; ",keyword)
        list4 = delist(" ",abstract)
        alist = list1+list2+list3+list4
        
        # lower case
        blist = lower(alist)
        # remove stopwords
        destop = remove_stop(blist)
        # remove punctuate
        final = depunct(destop)
        
        content["F3"]=final
        index[art_id]=content
    return index
    
def organisation_socket(corpus):
    index = corpus.copy()
    for art_id in index:
        content = corpus[art_id]
        address_str = content["C1"]
        address_list = match_addresses(address_str)
        org_list = []
        for address in address_list:
            address_parts = address.split(", ")
            org = address_parts[0]
            org_list.append(org)
        content["F2"]=org_list
        index[art_id]= content
    return index

from nltk.corpus import stopwords

def match_addresses(auth_str):
    
    result = re.findall(r"\[.+?\]", auth_str)
    for group in result:
        auth_str=auth_str.replace(group,"")
    address_list = auth_str.split("; ")

    new_list = []
    for address in address_list:
        address = address.replace("^\s+","")
        address = address.lstrip()  
        new_list.append(address)
        
    return new_list 

def remove_copyright(abstring):
    bstring = re.sub(r"\s\(C\)\s\d\d\d\d", "", abstring)
    cstring = re.sub(r"\sAll rights reserved.$",'',bstring)
    newstring = re.sub(r'\sElsevier (Inc.|Science Inc.|B.V.)','',cstring)
    return newstring
 
def delist(sep,astring):
    alist = astring.split(sep)
    return alist

def lower(alist):
    blist = []
    for el in alist:
        em = el.lower()
        blist.append(em)
    return blist

def remove_stop(alist):
    stop = stopwords.words('english')
    extended = ['paper','study','results','1','2','3','also','using','different','used',\
               'one','two','first','used','however']
    stop.extend(extended)
    filtered = [word for word in alist if word not in stop ]
    return filtered

def depunct(alist):
    blist = []
    for el in alist:
        em = re.sub('['+string.punctuation+']', '', el)
        
        if (len(em)> 0):
            blist.append(em)
    return blist

# don't ever count empty fields
def make_index(n,lbl,corpus):
    cntr = c.Counter()
    for art_id in corpus:
        content = corpus[art_id]
        alist= content[lbl]
        for el in alist:
            cntr[el]+=1

    atop = cntr.most_common(n+1)
    
    if ('' in atop):
        top=cntr.most_common(n+1)
        el,freq=zip(*top)
        elx = list(el)
        elx.remove('')
    else:
        top=cntr.most_common(n)
        el,freq=zip(*top)
        elx = list(el)     
 
    ids = list(range(len(elx)))
    index = dict(zip(elx,ids))
    return index

# takes a field name (i.e. "CR")
# takes a dictionary of items for indexing (i.e. cite_dict)
# takes a corpus
# NB This is using the whole citation as the key.
# However it returns a dictionary with article ids as keys, and index as values
def index_corpus(afield,adict,corpus):
    index = {}
    for art_id in corpus:
 
        content = corpus[art_id]
        alist = content[afield]
 
        for el in alist:
            if el in adict:
                if (art_id in index):
                    anum = adict[el]
                    alist = index[art_id]
                    alist.append(anum)
                    index[art_id]=alist
                else:
                    anum = adict[el]
                    index[art_id]=[anum]
 
                
    return index


def index_corpus_dual(afield,adict,corpus):
    dual = {}
    for art_id in corpus:
        content = corpus[art_id]
        cite_str = content[afield]
        citations = cite_str.split("; ")
        for cite in citations:
            if (cite in adict.keys()):
                aid = adict[cite]               
                if (aid in dual):
                    alist = dual[aid]
                    alist.append(art_id)
                    dual[aid]=alist
                else:
                    dual[aid]=[art_id]
    return dual

# n is the number of items in the index set
# adict is an indexing dictionary 
def compute_cooccur(n,adict):
 
    X = np.zeros((n,n))
    for item in adict:

        alist = adict[item]
 
        # WARNING: The sets may not be unique
   
        if (len(alist)>0):
            aset = set(alist)
            uniq = list(aset)
 
            if (len(uniq) > 1):
                C = itr.combinations(uniq,2)
            
                for pair in C:
                    a,b = pair
                    X[a,b]+=1
                    X[b,a]+=1


    return X

def create_label_dict(lbl,adict):
    
    label_dict = {}
    
    if (lbl=="F1"):
        label_dict = uniq_label_maker(adict)
    # must be F2 or F3
    else:
        for name in adict:
            label = label_maker("F2",name)
            value = adict[name]
            label_dict[label]=value

    return label_dict

def create_article_labels(corpus):
    n = len(corpus)
    keys = list(range(n))
    values = corpus.keys()
    article_labels = dict(zip(keys,values))
    return article_labels
# always F1
def uniq_label_maker(adict):
    bdict = {}
    cdict = {}

    for key in adict:
        parts = key.split(", ")
        name = parts[0]
        year = parts[1]
        name=name.upper()
        year=year.upper()
        lbl = name+" "+year
        lbl = lbl.replace(" ","_")
        lbl = lbl.replace(".","")
        lbl = lbl.replace("\"""","")
        if (lbl in bdict):
            bdict[lbl]+=1
        else:
            bdict[lbl]=0
 
    n=-1
    for key in bdict:

        cnt = bdict[key]
        for i in range(cnt+1):
            lbl = key+"_"+str(i)
            n+=1
            cdict[lbl]=n
 
    return cdict

# always F2
def label_maker(lbl, key):
    #key is a uni name
    label = key.replace(" ","_")
        
        
    return label

# maybe the full answer requires network X
def connected_core(label_dict,X):
    x, y = X.shape
    distinct = {}
    disconnect = {}
    for i in range(x):
        for j in range(y):
            num = X[i,j]
            if (num > 0):
                distinct[i]=0
                distinct[j]=0
    for i in range(x):
        if (i in distinct):
            next
        else:
            disconnect[i]=0
    return disconnect

#file name
#file labels
#co_occur
def write_dotnet(filename,label_dict,X):
    adict = invert_dict(label_dict)
    
    with open(filename, 'w') as f:
        n,m = X.shape
 
        f.write("*Vertices "+str(n)+"\n")
        for i in range(n):
            ii = i+1
            label = adict[i]
            f.write(str(ii)+" "+label+"\n")
        f.write("*Edges \n")
        for i in range(n-1):
            j = i+1
            for j in range(j,n):
                ii = i+1
                jj = j+1
                k = int(X[i,j])
                if (k > 0):
                    f.write(str(ii)+" "+str(jj)+" "+str(k)+"\n")





