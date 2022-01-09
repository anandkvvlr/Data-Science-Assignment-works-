import re
import random

### Question.1

w = "Grow Gratitude"
print(w)

## a) Access the letter 'G'
print("accesing 'G' ",w[0])    ##  accesing lesster 'G' by calling zeroth position of string 'w'
      ## or  ##
print( re.findall('G',w))

## b)	How do you find the length of the string?
print('length of the string "w" is',len(w))

## c)	Count how many times “G” is in the string.
print(len(re.findall('G',w))) 
      ## or  ##
print(w.count("G"))

########################################################################
#Quention2
import re
import random

s1 = "Being aware of a single shortcoming within yourself is far more useful than being aware of a thousand in someone else."
a = re.findall("\s",s1)   ## a=> set of total white spaces in s1  ## '\s'=> white space
print("number of characters in s1 is:",len(s1)-len(a))

########################################################################
#Quention3.Create a string "Idealistic as it may sound, altruism should be the driving force in business, not just competition and a desire for wealth"

#Solution

import re
import random

#Slicing
s2 = "Idealistic as it may sound, altruism should be the driving force in business, not just competition and a desire for wealth"
l = len(s2)

## print randomly any one of the character in the string        
def charac(s):
    chara=[]
    chara = [x for x in s2 if x!=" "]
    l= len(chara)
    return chara[random.randint(0,l)]
    

print(charac(s2))    

re.findall("^\s",s1)

print (s2[0:3]) #get the first three char
print (s2[:3]) #get the first three char
print (s2[-3:]) #get the last three char

###########################################################################
#Quention4.create a string "stay positive and optimistic". Now write a code to split on whitespace.
#Solution

import re
import random

# spliting
w3 = "stay positive and optimistic"
w3.split(' ') # Split on whitespace
# Startswith / Endswith
w3.startswith("H") # False
w3.endswith("d") # False
w3.endswith("c") #True
######################################################
#Quention5.Write a code to print " 🪐 " one hundred and eight times. (only in python)
#Solution

import re
import random

# repeat string 

for x in range(108):   # prints 108 times
    print("🪐")
    


#####################################################################
#Quention7.Create a string “Grow Gratitude” and write a code to replace “Grow” with “Growth of”
#Solution

import re
import random

# replacing

w4 = "Grow Gratitude"

w4.replace("Grow", "Growth of")


########################################
#Quention8.
#Solution

import re
import random

s3 = "elgnujehtotniffo deps mehtfohtoB .eerfnoilehttesotseporeht no dewangdnanar eh ,ylkciuQ .elbuortninoilehtdecitondnatsapdeklawesuomeht ,nooS .repmihwotdetratsdnatuotegotgnilggurts saw noilehT .eert a tsniagapumihdeityehT .mehthtiwnoilehtkootdnatserofehtotniemacsretnuhwef a ,yad enO .ogmihteldnaecnedifnocs’esuomeht ta dehgualnoilehT ”.emevasuoy fi yademosuoyotplehtaergfo eb lliw I ,uoyesimorp I“ .eerfmihtesotnoilehtdetseuqeryletarepsedesuomehtnehwesuomehttaeottuoba saw eH .yrgnaetiuqpuekow eh dna ,peels s’noilehtdebrutsidsihT .nufroftsujydobsihnwoddnapugninnurdetratsesuom a nehwelgnujehtnignipeelsecno saw noil A"
print (''.join(reversed(s3)))

#####################Thank you ########################################












































































