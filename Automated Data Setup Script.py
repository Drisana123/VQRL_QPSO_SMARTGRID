import pandapower.networks as pn
import pandas as pd
import numpy as np

def setup_research_data():
    # 1. Load the Grid
    net = pn.case118()
    print(f"Successfully loaded IEEE 118-bus system with {len(net.bus)} buses.")

    # 2. Simulate Renewable Stochasticity (If not using external CSV)
    # We create a 24-hour stochastic profile for the Wind Farm at Bus 10, Bus 100
    time_steps = 24
    base_wind_power = 100.0  # MW
    uncertainty = 0.2  # 20% variability
    
    # Generate Weibull-distributed wind profile
    wind_profile = base_wind_power * np.random.weibull(a=2, size=time_steps) * (1 + uncertainty)
    
    # 3. Create a Dataframe to store results
    res_data = pd.DataFrame({
        'Hour': range(time_steps),
        'Wind_MW': wind_profile
    })
    
    return net, res_data

# Run setup
grid, renewable_data = setup_research_data()
renewable_data.to_csv("renewable_penetration_profile.csv", index=False)
print("Data initialization complete. Profiles saved to 'renewable_penetration_profile.csv'.")