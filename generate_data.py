
import pandas as pd
import numpy as np

# Initialize random number generator
rng = np.random.default_rng(42)

# Number of registrations
n = 2000

# Generate synthetic event registration data
data = pd.DataFrame({
    "event_type": rng.choice(
        ["hackathon", "workshop", "cultural"], size=n
    ),
    "ticket_price": rng.choice(
        [0, 100, 300], size=n
    ),
    "days_before_event": rng.integers(
        1, 31, size=n
    ),
    "reminder_sent": rng.integers(
        0, 2, size=n
    ),
    "prior_attendance": rng.integers(
        0, 2, size=n
    ),
    "cancelled": rng.choice(
        [0, 1], size=n, p=[0.9, 0.1]
    )
})

# A reminder can only be opened if it was sent
data["reminder_opened"] = np.where(
    data["reminder_sent"] == 1,
    rng.integers(0, 2, size=n),
    0
)

# Calculate check-in probability
# Baseline check-in probability is 75%
checkin_probability = np.full(n, 0.75)

# Increase no-show probability if the reminder was not opened
checkin_probability -= np.where(
    data["reminder_opened"] == 0, 0.15, 0
)

# Increase no-show probability if registration was early
checkin_probability -= np.where(
    data["days_before_event"] > 14, 0.10, 0
)

# Increase no-show probability if there is no prior attendance
checkin_probability -= np.where(
    data["prior_attendance"] == 0, 0.15, 0
)

# Keep probabilities within a valid range
checkin_probability = np.clip(
    checkin_probability, 0.05, 0.95
)

# Generate check-in outcomes
data["checked_in"] = rng.binomial(
    1, checkin_probability
)

# Cancelled registrations cannot check in
data.loc[data["cancelled"] == 1, "checked_in"] = 0

# Save the dataset to CSV
data.to_csv("synthetic_data.csv", index=False)

# Display basic information
print("Synthetic event registration data generated successfully!")
print(f"Total registrations: {len(data)}")
print("\nFirst 5 rows:")
print(data.head())

print("\nDataset information:")
print(data.info())

print("\nCheck-in distribution:")
print(data["checked_in"].value_counts())

print("\nCancellation distribution:")
print(data["cancelled"].value_counts())

print("\nFile saved as synthetic_data.csv")
