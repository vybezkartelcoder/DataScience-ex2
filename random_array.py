import numpy as np

# Generate a random array of shape (3, 4) with values between 0 and 1
random_array = np.random.rand(3, 4)
print("Random array between 0 and 1:")
print(random_array)
print("\n")

# Generate a random array of shape (3, 4) with values from standard normal distribution
normal_array = np.random.randn(3, 4)
print("Random array from standard normal distribution:")
print(normal_array)
print("\n")

# Generate a random array of integers between 0 and 10
integer_array = np.random.randint(0, 10, size=(3, 4))
print("Random integer array between 0 and 10:")
print(integer_array)
print("\n")

# Generate a random array with specific mean and standard deviation
mean = 5
std = 2
custom_array = np.random.normal(mean, std, size=(3, 4))
print(f"Random array with mean={mean} and std={std}:")
print(custom_array) 