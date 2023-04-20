import numpy as np
import matplotlib.pyplot as plt
 
# set width of bar
barWidth = 0.4
fig = plt.subplots(figsize =(12, 8))
 
# set height of bar
MSE = [2.9379337266458014, 2.4604027778, 2.3371768, 2.516066584, 2.401490026, 2.288296881]
SSIM = [0.9958493133915519 , 0.9965, 0.996647183248, 0.9964553574, 0.9965550889, 0.996681867]

# Set position of bar on X axis
br1 = np.arange(len(MSE))
 
# Make the plot
plt.bar(br1, MSE, color ='teal', width = barWidth,
        edgecolor ='grey', label ='Masked-MSE')
 
# Adding Xticks
plt.xlabel('Model', fontweight ='bold', fontsize = 15)
plt.ylabel('Test Masked MSE Loss', fontweight ='bold', fontsize = 15)
plt.xticks([r for r in range(len(MSE))],
        ['HiFiC Low', 'Masked MSE', 'k_P = 100', 'k_P = 75', 'k_P = 50',
            'k_M = 5\nk_P = 100'])
 
plt.legend()
plt.savefig("./mseloss.png", dpi=150, format='png', pad_inches=0.1)
