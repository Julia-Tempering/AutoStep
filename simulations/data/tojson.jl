using CSV, DataFrames, JSON

# Read CSV file into a DataFrame
df = CSV.read("icml2025/data/prostate.csv", DataFrame)

# Extract y (first column)
y = df[:, 1]

# Extract x (remaining columns) as a list of arrays
x = [collect(row) for row in eachrow(df[:, 2:end])]

# Compute dimensions
n = length(y)   # Number of observations
d = size(df, 2) - 1  # Number of predictors (total columns - 1)

# Create a dictionary for JSON output
data_dict = Dict(
    "n" => n,
    "d" => d,
    "y" => y,
    "x" => x
)

# Save to a JSON file
open("icml2025/data/prostate.json", "w") do f
    JSON.print(f, data_dict, 4)  # Pretty-print with indentation
end