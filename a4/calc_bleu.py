import nltk
import numpy as np
import pdb

r1 = "resources have to be sufficient and they have to be predictable"
r2 = "adequate and predictable resources are required"

c1 = "there is a need for adequate and predictable resources"
c2 = "resources be sufficient and predictable to"

#  Human sesne says c1 is better matching to r1, but without r1 c_2 will probably be better

def main():
    lam = [0.5,0.5] # lambda weighting values
   
    print("\npart i")
    print("-----------------------------")
    lam = [0.5,0.5] # lambda weighting values
    r = [r1,r2] # reference sentences, assume sorted shortest to longest
    print("c1: ")
    calc_bleu(c1, lam, r)
    #print("c2: ")
    #calc_bleu(c2, lam, r)


    ## part i: calculate bleu scores without r1
    #print("\npart ii")
    #print("-----------------------------")
    #r = [r1] # reference sentences, assume sorted shortest to longest
    #print("c1: ")
    #calc_bleu(c1, lam, r)
    #print("c2: ")
    #calc_bleu(c2, lam, r)
    
    # compare to true values 
    # nltk.translate.bleu_score.sentence_bleu([r1.split(), r2.split()], c1.split(), lam)  ~ 0.75

def calc_bleu(c, lam, r):
    ''' calculate the bleu value 

        @param c candidate sentence (string)
        @param lam list of lambda values staring at ngram=1, e.g. [0.5,0.5]
                     also tells how many ngram models to calculate, e.g. n=1, n=2
        @param r list of reference sentences, list(strings)

        @return bleu score calculated as described in A4 handout '''
        
    p_n = [] # Initalize p storage
    for n in range(1,len(lam)+1) :
        n_gram = list(nltk.ngrams(c.split(),n))
        num = 0
        denom = 0
        print(n_gram)
        for gram in set(n_gram) :
            print(gram, n_gram.count(gram))
            # Count number of matches to sentence
            num += min([max([list(nltk.ngrams(r_n.split(), n)).count(gram) for r_n in r])] + [n_gram.count(gram)])
            denom += n_gram.count(gram)
            print(denom)
        p_n.append(num/denom)

    # Brevity penalty
    val, idx = min((np.abs(len(r_n.split())-len(c.split())), idx) for (idx,r_n) in enumerate(r))
    bp = min(1, np.exp(1 - len(r[idx].split())/len(c.split())))
    
    # weighted average over lam values
    bleu = bp * np.exp(np.array(lam) @ np.array(np.log(p_n)))

    # Print array of intermeidary calculations
    print("p values: ", p_n)
    print("bp: ", bp)
    print("len(c): ", len(c.split()))
    print("len(r): ", len(r[idx].split()))
    print("bleu: ", bleu)

    return bleu 

if __name__ == '__main__':
    main()
