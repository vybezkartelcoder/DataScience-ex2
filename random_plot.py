import numpy as np
import matplotlib.pyplot as plt

# Generate random numbers
# Δημιουργία τυχαίων αριθμών
random_numbers = np.random.randn(100)

# Create a simple plot
# Δημιουργία απλού γραφήματος
plt.figure(figsize=(10, 6))
plt.plot(random_numbers, '-o', markersize=4)
plt.title('Random Numbers Plot / Γράφημα Τυχαίων Αριθμών')
plt.xlabel('Index / Δείκτης')
plt.ylabel('Value / Τιμή')
plt.grid(True)

# Add a horizontal line at y=0
# Προσθήκη οριζόντιας γραμμής στο y=0
plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)

# Save the plot
# Αποθήκευση του γραφήματος
plt.savefig('random_plot.png')

# Show the plot
# Εμφάνιση του γραφήματος
plt.show() 