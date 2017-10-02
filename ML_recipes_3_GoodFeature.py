
# coding: utf-8

# In[6]:


import numpy as np
import matplotlib.pyplot as plt
labs=500
greyhounds=500

grey_height=28+4*np.random.randn(greyhounds)
lab_height=28+4*np.random.randn(labs)

plt.hist([grey_height,lab_height],stacked=True,color=['r','b'])
plt.show()


# In[ ]:





# In[ ]:




