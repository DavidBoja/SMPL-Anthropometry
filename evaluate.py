
def evaluate_mae(gt_measurements,estim_measurements):
    '''
    Compare two sets of measurements - given as dicts - by finding
    the mean absolute error (MAE) of each measurement.
    :param gt_measurement: dict of {measurement:value} pairs
    :param estim_measurements: dict of {measurement:value} pairs

    Returns
    :param errors: dict of {measurement:value} pairs of measurements
                    that are both in gt_measurement and estim_measurements
                    where value corresponds to the mean absoulte error (MAE)
                    in cm
    '''

    MAE = {}

    for m_name, m_value in gt_measurements.items():
        if m_name in estim_measurements.keys():
            error = abs(m_value - estim_measurements[m_name])
            MAE[m_name] = error

    if MAE == {}:
        print("Measurement dicts do not have any matching measurements!")
        print("Returning empty dict!")

    return MAE


if __name__ == "__main__":

    import torch
    import pandas as pd
    from measure import MeasureSMPL
    from measurement_definitions import MeasurementDefinitions

    smpl_path = "/SMPL-Anthropometry/data/SMPL"
    
    measurer1 = MeasureSMPL(smpl_path=smpl_path)
    betas1 = torch.empty((1,10)).normal_(mean=0,std=1)
    measurer1.from_smpl(gender="MALE", shape=betas1)

    measurer2 = MeasureSMPL(smpl_path=smpl_path)
    betas2 = torch.empty((1,10)).normal_(mean=0,std=1)
    measurer2.from_smpl(gender="MALE", shape=betas2)


    measurement_names = MeasurementDefinitions.possible_measurements
    measurer1.measure(measurement_names)
    measurer2.measure(measurement_names)

    
    MAE = evaluate_mae(measurer1.measurements,measurer2.measurements)    
    mae_table = pd.DataFrame({"Measurement":MAE.keys(),
                              "MAE(cm)": MAE.values()})
    print(mae_table)