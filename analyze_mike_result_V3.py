# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 00:09:58 2024

@author: mdahal
"""

### This script will compute performance metrices, create the comparison graphs for the calibrated basins given various inputs

        ## The graph compares model results with flow meter and also include rainfall hyetograph.
        ## The graph also shows the performance metrices.

## The two inputs are:
    # inputs.csv



# event_dir_file : 	This is the directory with file name that contains the event_list csv file

# model_dir : This is the model directory where the model res1d file is located. This is where the images will be saved as well.

# Flow_dir : This is the flow directory including the file name. 

# res1d_file : Name of the res1d file that has the final calibrated results. Note: Don’t forget the file extension.

# scenario_name : Name of Scenario

# c_name : Name of the chromosome in GA that is supposed to have best calibrated results.

# rain_dir_file: Rain directory with file name.

# rain_width : This is the width of hyetograph. The unit is hour.

### This version will also be able to compute the Average Daily Flow and make average daily flow graph
    

#######################
### Assumptions

    ## The rain hyetograph with is in hour

## Flow data is already processed.. using flow processor tool
    ## 
    ## Flow data will be converted to mgd from m3/sec

## Event list file
    ## Date format : "MM/DD/YYYY" ( "%m/%d/%Y" )


## Result file
    ## Assuming the best fit is in the result "CHR001"
    
    
### Packages
import os
import glob
import subprocess
import pandas as pd
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
from plotnine import *
import math
import matplotlib.dates as mdates
from mikeio1d import Res1D
from mikeio1d.res1d import QueryDataCatchment
import openpyxl
import argparse
import sqlite3
import warnings
warnings.filterwarnings('ignore')

################### Functions ##################################################
##################################################################################

########### Get optimized parameters #############

def get_optimized_parameters(sqfile_dir, calibrated_parameters):

    '''
    Given the directory of sqlite file, the calibrated parameters are extracted.
     
    sqfile_dir: Full directory of model sqlite file
    
    calibrated_parameters: Dataframe of calibrated parameters.

    '''

    
    #sqfile_dir = model_dir+"\\"+sqlite_file
    conn = sqlite3.connect(sqfile_dir)
    cursor = conn.cursor()
    df = pd.read_sql_query("SELECT * FROM msm_Catchment", conn)
    df2 = pd.read_sql_query("SELECT * FROM msm_HParRDII", conn)
    ### catchment information #####################################
    aflow = df[df['muid']==c_name]["addflow"].iloc[0] * 22.83  ### convert to mgd from cumec
    rdii_area = df[df['muid']==c_name]['rdiiarea'].iloc[0]  * 100  ### convert to percent from decimal
    a_iarea = df[df['muid']==c_name]['modelaimparea'].iloc[0] * 100 ### convert to percent from decimal
    a_ctime = df[df['muid']==c_name]['modelaconctime'].iloc[0] / 60  ### convert to minute from seconds
    modelbaiflat = df[df['muid']==c_name]["modelbaiflat"].iloc[0] * 100 ## convert to percent
    modelbmiflat = df[df['muid']==c_name]["modelbmiflat"].iloc[0] * 1
    ################################################################
    
    #### Rdii information ##########################################
    umax = df2[df2['muid']==RDII_name]["umax"].iloc[0] * 39.3701 ### convert to in from m
    lmax = df2[df2['muid']==RDII_name]["lmax"].iloc[0] * 39.3701  ### convert to in from m
    cqof = df2[df2['muid']==RDII_name]["cqof"].iloc[0] 
    ck = df2[df2['muid']==RDII_name]["ck"].iloc[0]
    ckif = df2[df2['muid']==RDII_name]["ckif"].iloc[0]
    ckbf = df2[df2['muid']==RDII_name]["ckbf"].iloc[0]
    gw_carea = df2[df2['muid']==RDII_name]["gwcarea"].iloc[0]
    tof = df2[df2['muid']==RDII_name]["tof"].iloc[0]
    tif = df2[df2['muid']==RDII_name]["tif"].iloc[0]
    tg = df2[df2['muid']==RDII_name]["tg"].iloc[0]
    #################################################################
    
    par_dict = {"aflow":aflow, "rdii_area":rdii_area, "a_iarea":a_iarea, "a_ctime":a_ctime,
                "modelbaiflat":modelbaiflat, "modelbmiflat":modelbmiflat, "umax":umax,"lmax":lmax,
                "cqof":cqof, "ck":ck, "ckif":ckif, "ckbf":ckbf,"gw_carea":gw_carea,
                "tof":tof,"tif":tof,"tg":tg}
    optimized_parameters = pd.DataFrame(par_dict.items(), columns= ['Parameter','Value'])
    #optimized_parameters
    final_optimiz_params = pd.merge(left = calibrated_parameters, right = optimized_parameters, how='inner', left_on= "Calibrated Parameters", right_on = "Parameter")

    return final_optimiz_params




############ Compute performance metrics #######################

def compute_metrics(flow_data_filt, qdf_filt, save_int):

    '''
        Provided the data frame for flow data and model result, this function will compute the performance metrics of the data.
        
        flow_data_filt: Flow data that is filtered for a given event
        qdf_filt: model result that is extracted from Res1d file and filtered for a given event
        save_int: str; should the comparison graph between raw and interpolated data be saved (TRUE OR FALSE)
        
        #### This will also interpolate the model result if the model result time series dont align with flow_data time series.
        
        output:
            ldf3: A dataframe with measured and modeled data after aligning the time and interpolating. 
                    This is the data frame used for calculating metrics.
           Performance metrics: Nash coefficient, RMSE, Relative Q, Relative 2day Vol
                   
    '''


    
    print("Comparing model result with flow meter data...")
    ### Merging using time index ###
    flow_data_filt_ind = flow_data_filt.set_index('Date&Time PST/PDT')
    qdf_filt_ind = qdf_filt.set_index('Time')
    ldf = pd.merge(flow_data_filt_ind,qdf_filt_ind, left_index = True, right_index = True, how = 'outer')
    #data_merge = pd.merge(flow_data_filt,qdf_filt, left_on = 'Date&Time PST/PDT', right_on = 'Time', how = 'outer')
    Nas = ldf[ldf['Data'].isna()]['Data']
    if(len(Nas)>0):
        print("Modeled data need interpolation. This could happen if the timeseries in flow meter and model result dont align or are of different resolution.")
        print("Number of data points needing interpolation: {0}".format(len(Nas)))
        print("Conducting Linear Interpolation.")
        ldf['Interpolated_Data'] = ldf['Data'].interpolate(method = 'linear')
        ##### Plotting interpolation quality ########
        
        raw_data = ldf[ldf['Data'].notna()]
        fig, ax = plt.subplots(figsize =(10,6))
        ax.plot(raw_data['Data'], label = 'Raw_data',linewidth = 5)
        #ax.plot(raw_data['Data'],  'bo', label = 'Raw_data')
        ax.plot(ldf['Interpolated_Data'], label = 'Interpolated_data')
        ax.set_title("{0}: Raw model result vs interpolated data for aligning with flow meter time series".format(event_name))
        ax.legend()
        ## saving interpolated graph
        print("Saving interpolated data vs raw data figure in model directory.")
        if (save_int == "TRUE"):
            plt.savefig(model_dir+"\\Model_result_Interpolated_VS_Raw_"+event_name+".png")
        #######################################################
        ldf2 = ldf.loc[flow_data_filt_ind.index]
        ldf3 = ldf2[["I/I mgd","Interpolated_Data"]]
        ldf3.columns = ['Measured','Modeled']
    else:
        ldf2 = ldf.loc[flow_data_filt_ind.index]
        ldf3 = ldf2[["I/I mgd","Data"]]
        ldf3.columns = ['Measured','Modeled']
        print("Interpolation not needed.") 
    print("Computing performance metrices...")
    
    #### Compute metrices #############
    ldf4 = ldf3.reset_index()
    
    #### Nash ###########
    q_obs_mean = ldf4['Measured'].mean()
    ldf4['q_error'] =  pow((ldf4['Measured'] - ldf4['Modeled']),2)
    ldf4['q_mean_error'] =  pow((ldf4['Measured'] - q_obs_mean),2)
    nash_nominator = ldf4['q_error'].sum()
    nash_denominator = ldf4['q_mean_error'].sum()
    Nash = round(1 - (nash_nominator/nash_denominator),2)
    ##### RMSE ###############
    RMSE = round(math.sqrt(nash_nominator/(len(ldf4['q_error']))),2)
    ####### Qpeak ##################
    Relq = round((ldf4['Modeled'].max())/(ldf4['Measured'].max()),2)
    ######## Rel_Vol2d_err ##############################################
    peak_measured_time =ldf4[ldf4['Measured']==ldf4['Measured'].max()]['Date&Time PST/PDT']
    first_dt = (peak_measured_time - timedelta(days = 1)).iloc[0]
    last_dt = (peak_measured_time + timedelta(days = 1)).iloc[0]
    mask_48hr = (ldf4['Date&Time PST/PDT']>=first_dt) & (ldf4['Date&Time PST/PDT']<=last_dt) 
    ldf4_48hr = ldf4[mask_48hr]
    
    #### Measured total volume ###
    Vol_2d_modeled = ldf4_48hr['Modeled'].sum()
    Vol_2d_measured = ldf4_48hr['Measured'].sum()
    #diff_vol2d = Vol_2d_measured - 
    Relvol = round((Vol_2d_modeled/Vol_2d_measured),2)

    return [ldf3, Nash, RMSE, Relq, Relvol]


##############################################################################################
######## Create plots ##############################
def create_graph(model_date, model_y, flow_met_date, flow_met_y, diurn_date, diurn_y, rain_date, rain_y,
                 day_ints,max_mgd, Nash, RMSE, Relq, Relvol, max_rain,width_fact,rain_width,
                 scenario_name,event_name,model_dir):
    """
    This function will generate comparision graphs between model and flow meter and insert various performance metrices

    Parameters
    ----------
    model_date : df 
        x axis date of model result
    model_y : df
        y - axis flow of model result
    flow_met_date : df
        x axis date of flow meter data
    flow_met_y : df
        y axis flow of flow meter data
    diurn_date : df
        x axis date of diurnal data
    diurn_y : df
        y axis flow of diurnal data
    rain_date : df
        x axis date of rain data
    rain_y : df
        y axis values of rain data
    day_ints : int 
        value for x-axis label intervals
    max_mgd : int
        ylimit max for flow
    Nash : int
        Nash sutcliffe efficiency
    RMSE : int
        rootmeansquare error
    Relq : int
        Relative q peak.
    Relvol: int
        Relative volume
    max_rain : int
        ylimit max for rain
    width_fact: int
        this is the factor used for adjusting the bar graph width properly
    rain_width : int
        bar width for hyetograph
    scenario_name: str
        Name of the scenario 
    event_name : str
        string of event name
    model_dir : str
        model directory which is used for saving the plots
    basin_name: str
        This is the name of the basin.

    Returns
    -------
    Comparison plots are prepared and saved in the model directory with proper name.

    """
    fig, ax = plt.subplots(figsize = (14, 8))
    ax2 = ax.twinx()
    #ax.plot(qdf_filt["Time"], qdf_filt["Data"], label = real_name, color = "red")
    ax.plot(model_date, model_y, label = scenario_name, color = "red")
    ax.plot(flow_met_date, flow_met_y, label = (scenario_name+"_mtr"), color = "darkblue")
    ax.plot(diurn_date, diurn_y, label = "Diurnal", color = "chartreuse")
    ax.xaxis.set_major_locator(mdates.DayLocator(interval = day_ints))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d/%Y"))
    ax.set_ylabel("Flow (mgd)", fontsize = 12)
    ax.set_ylim(0, max_mgd)
    #### Adding text ####
    #ax.text(1.06, 1.0,event_name,transform=ax.transAxes, size = 14)
    text = "NASH: "+str(Nash)+"\n"+"RMSE: "+str(RMSE)+"\n"+"Rel Qpk: "+str(Relq)+"\n"+"Rel 2d Vol: "+str(Relvol)
    ax.text(1.1, 0.325,"Goodness of Fit Metrics",transform=ax.transAxes, fontweight = "semibold", size = 12)
    ax.text(1.1, 0.2,text,transform=ax.transAxes, size = 14)
    
    
    #### Plotting hyetograph
    #ax2.plot(rain_data_filt['datetime'], rain_data_filt['rain'], color = "lightblue", label = "Precip")
    ax2.bar(rain_date, rain_y, edgecolor = "skyblue", label = "Precip", width = width_fact, fill = False)
    ax2.set_ylim(0,max_rain)
    ax2.invert_yaxis()
    ax2.set_ylabel(str(rain_width)+"-hr rainfall (in)", fontsize = 12)
    #ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    # ask matplotlib for the plotted objects and their labels
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2,bbox_to_anchor=(1.06, 0.9), loc="upper left", fontsize = 12, title = event_name,title_fontproperties = {'weight':"bold", "size":14})
    fig.tight_layout()
    rect = plt.Rectangle((0.005,0.005),0.99,0.99, transform = fig.transFigure, linewidth = 1, edgecolor = "black", facecolor = "none")
    fig.patches.append(rect)
    #fig.subplots_adjust(right=0.2)
    plt.savefig(model_dir+"\\"+event_name+"_" + basin_name + ".png")
    

###################################################################################
###############################################################################################
### Update value based on header name, basin row, and the value itself using openpyxl

def update_val(wb_st, header_name, header_name_row, basin_row, value):

    '''
    library required: openpyxl
    Given the name of the workbook sheet object, header, basin row number and value itself, this function will fill the cell in the excel workbook object.

    wb_st: Workbook sheet object (using the openpyxl object)
    header_name_row: The row number that contains the header
    header_name: Name of the header colomn
    basin_row: Basin row number
    value: Value that needs to be filled

    output: The wb_st object is updated with the value.
    
    '''
    for cells in wb_st[header_name_row]:
        if (cells.value == header_name):
            par_no = cells.column_letter
            #print('parameter: {0} and column number: {1}'.format(mtnm,par_no))
            cellno = par_no+str(basin_row)
            #print(cellno)
            parvalue = value
            # print(parvalue)
            wb_st[cellno] = parvalue
#################################################################



#os.getwd()
## Get inputs
parser = argparse.ArgumentParser(prog = 'Create Graph', description = 'The program uses input.csv file to create various graphs.')
parser.add_argument('filename', help = 'Name of the directory that contains the input.csv file')
args = parser.parse_args()
input_data_dir = args.filename

### Get major vars
input_dir_file = input_data_dir #r"M:\user\dahal\CSI_documentation\Mainfiles\Test_subject_2\inputs.csv"
input_data = pd.read_csv(input_dir_file)

### Get Event list
#event_dir_file = r"M:\user\dahal\CSI_documentation\event_list.csv"
print("Getting events.")
evtnparams_dir_file = input_data['Input'].loc[7]
evtnparams_data = pd.read_csv(evtnparams_dir_file)
calibrated_parameters = pd.DataFrame(evtnparams_data['Calibrated Parameters'])
event_data = evtnparams_data[['Start Date','End Date','Event']]
event_data = event_data[event_data['Event'].notna()]
event_data = event_data[event_data['Event']!=' ']

adf_event_data = evtnparams_data[['ADF_start', 'ADF_end', 'ADF_event']]
adf_event_data = adf_event_data[adf_event_data['ADF_event'].notna()]
adf_event_data = adf_event_data[adf_event_data['ADF_event']!=' ']
### Model directory ####################################

### This is where the sqlite and res1d file is stored
## AND the comparison images will be stored...
model_dir = input_data['Input'].loc[1]
res1d_file = input_data['Input'].loc[3]
resultdir = model_dir + "\\" + res1d_file
sqlite_file = input_data['Input'].loc[2]
sqfiledir = model_dir+"\\"+sqlite_file

### Catchment name 
scenario_name = input_data['Input'].loc[4]
c_name = input_data['Input'].loc[5]
basin_name = input_data['Input'].loc[6]
RDII_name = "RDII_"+c_name 

print("Extracting model results.")
### Extract flow data for all event 
res1d_catchment2 = Res1D(resultdir)
q1 = QueryDataCatchment("TotalRunOff",c_name)
qdf = res1d_catchment2.read(q1).reset_index()
qdf.columns = ["Time", "Data"]
### converting to mgd
qdf['Data'] = qdf['Data'] * 22.824465227271

### ADF ############
qdf_adf = qdf.groupby(pd.Grouper(key = "Time", freq = "1D" )).mean().reset_index()
#######################################################

### Rain gauge directory ########################################################################
print("Extracting rain and flow data.")
##### selecting data based on colomn number to give the flexibility ############
rain_dir_file = input_data['Input'].loc[12]
rain_date_ind = int(input_data['Input'].loc[13]) - 1 ### subtracted 1 because the colomn values are not starting from 0
rain_val_ind = int(input_data['Input'].loc[14]) - 1 ### substracted 1 because the colomn values are not starting from 0
rain_data_a = pd.read_csv(rain_dir_file, index_col = False)
rain_data = rain_data_a.iloc[:,[rain_date_ind,rain_val_ind]]
rain_data.columns = ["datetime", "rain"]

### convert datetime colomn to datetime object
rain_data['datetime'] = pd.to_datetime(rain_data['datetime'] )
#rain_data[1] = pd.to_datetime(rain_data[1] )
### What is the rainfall hyetograph width 
rain_width = float(input_data['Input'].loc[16]) #hr

##### Aggregate the rainfall
rain_data_agg = rain_data.groupby(pd.Grouper(key = "datetime",freq = (str(rain_width)+"h")))['rain'].sum().reset_index()

### Daily aggregate
rain_data_daily = rain_data.groupby(pd.Grouper(key = "datetime",freq = (str(24)+"h")))['rain'].sum().reset_index()

#################################################################################################################################

### Flow meter directory #######################################################################
### Get the processed flow
### This should have separate colomn of diurnal and Flow
flow_dir_file = input_data['Input'].loc[8]
flow_date_ind = int(input_data['Input'].loc[9]) - 1 ### subtracted 1 because the colomn values are not starting from 0
flow_diurn_ind = int(input_data['Input'].loc[10]) - 1 ### substracted 1 because the colomn values are not starting from 0
flow_ini_ind = int(input_data['Input'].loc[11]) - 1 ### substracted 1 because the colomn values are not starting from 0

flow_data_a = pd.read_csv(flow_dir_file)
flow_data = flow_data_a.iloc[:,[flow_date_ind,flow_diurn_ind, flow_ini_ind]]
flow_data.columns = ['Date&Time PST/PDT','Diurnal mgd','I/I mgd']
##########################################################################


### convert datetime colomn to datetime object
flow_data['Date&Time PST/PDT'] = pd.to_datetime(flow_data['Date&Time PST/PDT'] )

### Remove empty rows from the data

flow_data_2 = flow_data[flow_data['I/I mgd'] != ' '].copy()
flow_data_2['I/I mgd'] = flow_data_2['I/I mgd'].astype(float)

### Diurnal

#### Remove empty rows from the data
#flow_data_3 = flow_data[flow_data['Diurnal mgd'] != ' '].copy()
flow_data_2['Diurnal mgd'] = flow_data_2['Diurnal mgd'].astype(float)

#### ADF #######
flow_data_adf = flow_data_2.groupby(pd.Grouper(key = "Date&Time PST/PDT", freq = "1D" )).mean().reset_index()

#########################################################################################################

###### Other analysis parameters #########
save_interpolated_graph = input_data['Input'].loc[17] #"TRUE"
update_main_excel = input_data['Input'].loc[18] 
modeler_name = input_data['Input'].loc[0] 
main_excel_dir = input_data['Input'].loc[15] 

### 
metrics_list = list()
################################################################################

### Loop for each short events ####################

for i in range(0, len(event_data)):
    #### event information #####
    e1 = event_data.loc[i]
    event_name = e1['Event']
    
    print("\n Working on {0}".format(event_name))
    
    start_date = e1['Start Date']
    end_date = e1['End Date']
    start_dt = datetime.strptime(start_date, "%m/%d/%Y")
    end_dt = datetime.strptime(end_date, "%m/%d/%Y")
    interv = end_dt - start_dt
    int_sec = interv.total_seconds()
    int_day = int_sec/(3600*24)
    ######################
    
    
    ### Filter model 
    m1 = (qdf["Time"]>=start_date)&(qdf["Time"]<=end_date)
    qdf_filt = qdf[m1].reset_index(drop = True)
    
    ### Filter rain
    m2 = (rain_data_agg["datetime"]>=start_date)&(rain_data_agg["datetime"]<=end_date)
    rain_data_filt = rain_data_agg[m2].reset_index(drop = True)
    
    ### Filter flow meter data
    m3 = (flow_data_2['Date&Time PST/PDT']>=start_date)&(flow_data_2['Date&Time PST/PDT']<=end_date)
    flow_data_filt = flow_data_2[m3].reset_index(drop = True)
    
    #### X-axis Intervals for plots
    if int_day <= 10:
        day_ints = 1
    elif (int_day > 10) & (int_day <=  20):
        day_ints = 2
    elif (int_day > 20) & (int_day <= 30):
        day_ints = 3
    elif (int_day > 30) & (int_day <= 40):
        day_ints = 4
    else:
        day_ints = 5
    ####################
    
    ### factor for bar width
    ### 0.3 bar width worked based for 8 hour graph

    fact = rain_width/8
    width_fact = 0.3*fact
    
    ### Ylimits for flow
    
    max_mgd = pd.DataFrame([qdf_filt['Data'].max(), flow_data_filt["I/I mgd"].max()]).max()[0]
    max_mgd = math.ceil(max_mgd + 1)  
    
    
    ### Rain 
    max_rain = rain_data_filt['rain'].max()
    max_rain = math.ceil(max_rain * 3)
    
    ### Extract the NASH#, Qpeak, Volume
    ############################### Calculating Metrices #################

    met_list = compute_metrics(flow_data_filt, qdf_filt, save_interpolated_graph)
    
    ldf3 = met_list[0].reset_index()
    Nash = met_list[1]
    RMSE = met_list[2]
    Relq = met_list[3]
    Relvol = met_list[4]

    ##################################################################################

    
    ###############################Plot the graph ###################################
    
    print("Creating graphs!")
    
    #### Plot the graph for each events ###
    
    create_graph(ldf3["Date&Time PST/PDT"],ldf3["Modeled"],flow_data_filt['Date&Time PST/PDT'],flow_data_filt["I/I mgd"],flow_data_filt['Date&Time PST/PDT'], flow_data_filt["Diurnal mgd"],
                 rain_data_filt['datetime'],rain_data_filt['rain'],
                 day_ints,max_mgd, Nash, RMSE, Relq, Relvol, max_rain,width_fact,rain_width,
                 scenario_name,event_name,model_dir)
    
    ##################################################

    
    print("{0} completed! \n".format(event_name))
    
    list_res = [event_name,Nash,RMSE, Relq, Relvol]
    metrics_list.append(list_res)
    


##### Loop for ADF events #############################################
#######################################################################

for i in range(0, len(adf_event_data)):
    #### event information #####
    e1 = adf_event_data.loc[i]
    event_name = e1['ADF_event']
    
    print("\n Working on {0}".format(event_name))
    
    start_date = e1['ADF_start']
    end_date = e1['ADF_end']
    start_dt = datetime.strptime(start_date, "%m/%d/%Y")
    end_dt = datetime.strptime(end_date, "%m/%d/%Y")
    interv = end_dt - start_dt
    int_sec = interv.total_seconds()
    int_day = int_sec/(3600*24)
    ######################
    
    
    ### Filter model 
    m1 = (qdf_adf["Time"]>=start_date)&(qdf_adf["Time"]<=end_date)
    qdf_adf_filt = qdf_adf[m1].reset_index(drop = True)
    
    ### Filter rain
    m2 = (rain_data_daily["datetime"]>=start_date)&(rain_data_daily["datetime"]<=end_date)
    rain_adf_filt = rain_data_daily[m2].reset_index(drop = True)
    
    ### Filter flow meter data
    m3 = (flow_data_adf['Date&Time PST/PDT']>=start_date)&(flow_data_adf['Date&Time PST/PDT']<=end_date)
    flow_adf_filt = flow_data_adf[m3].reset_index(drop = True)
    
    #### X-axis Intervals for plots
    if int_day <= 30:
        day_ints = 5
    elif (int_day > 30) & (int_day <=  60):
        day_ints = 10
    elif (int_day > 60) & (int_day <= 80):
        day_ints = 30
    elif (int_day > 80) & (int_day <= 100):
        day_ints = 40
    else:
        day_ints = 60
    ####################
    
    ### factor for bar width
    ### 0.3 bar width worked best for 8 hour graph
    rain_width_2 = 24 #hr. assuming 24 hr bin width
    fact = rain_width_2/8 ## basically using unitary method for different rain_width value. 24 being for 24 hr
    width_fact = 0.3*fact
    
    ### Ylimits for flow
    
    max_mgd = pd.DataFrame([qdf_adf_filt['Data'].max(), flow_adf_filt["I/I mgd"].max()]).max()[0]
    max_mgd = math.ceil(max_mgd + 1)  
    
    
    ### Rain 
    max_rain = rain_adf_filt['rain'].max()
    max_rain = math.ceil(max_rain * 3)
    
    ### Extract the NASH#, Qpeak, Volume
    ############################### Calculating Metrices #################

    met_list2 = compute_metrics(flow_adf_filt, qdf_adf_filt, save_interpolated_graph)
    
    ldf3 = met_list2[0].reset_index()
    Nash = met_list2[1]
    RMSE = met_list2[2]
    Relq = met_list2[3]
    Relvol = met_list2[4]

    ##################################################################################

    
    ###############################Plot the graph ###################################
    
    print("Creating graphs!")
    
    #### Plot the graph for each events ###
    
    create_graph(ldf3["Date&Time PST/PDT"],ldf3["Modeled"],flow_adf_filt['Date&Time PST/PDT'],flow_adf_filt["I/I mgd"],flow_adf_filt['Date&Time PST/PDT'], flow_adf_filt["Diurnal mgd"],
                 rain_adf_filt['datetime'],rain_adf_filt['rain'],
                 60,max_mgd, Nash, RMSE, Relq, Relvol, max_rain,width_fact,rain_width_2,
                 scenario_name,event_name,model_dir)
    
    ##################################################

    
    print("{0} completed! \n".format(event_name))
    
    list_res = [event_name,Nash,RMSE, Relq, Relvol]
    metrics_list.append(list_res)
    
######################################################################

########################################################################################
############ Updating the main directory
met_df = pd.DataFrame(metrics_list, columns = ["Event name","Nash","RMSE","Relative Q","Relative Vol"])

############### Get the optimized parameters after calibration ###############################
optimized_parm = get_optimized_parameters(sqfiledir , calibrated_parameters)
optimized_parm['Value'] = round(optimized_parm['Value'],3)
print(met_df)
print(optimized_parm)
#met_df.to_csv(model_dir+"\\performance_metrics.txt", index=None)
#optimized_parm.to_csv(model_dir+"\\optimized_parameters.txt", index = None)

############ Updating excel file ##################
####################################################


if update_main_excel=="TRUE":
    print("Updating main excel file.")
    
    Main_excel_file = main_excel_dir
    
    #### Write to excel #####
    wb_obj = openpyxl.load_workbook(Main_excel_file)
    #sheet = wb_obj.active
    wb_st = wb_obj['Master']
    

    my_opt_par = optimized_parm
    my_perf_metric = met_df
    
    Nash_mean = round(my_perf_metric['Nash'].mean(),2)
    RelQ_mean = round(my_perf_metric['Relative Q'].mean(),2)
    RelVol_mean = round(my_perf_metric['Relative Vol'].mean(),2)
    
    basin_name_colomn = 'A'
    header_name_row = 3
    
    #### Get row number for the basin #######
    
    for cell in wb_st[basin_name_colomn]:
        if(cell.value == basin_name): #We need to check that the cell is not empty.
        #if 'Table' in cell.value: #Check if the value of the cell contains the text 'Table'
            print('Found basin with name: {0} at row: {1}.'.format(cell.value,cell.row))
            basin_row = cell.row

    #############################################################
    
    ######### Update calibrated parameters ###########

    for i in list(my_opt_par['Calibrated Parameters']):
    
        opt_par_value = my_opt_par[my_opt_par['Calibrated Parameters']==i]['Value'].iloc[0]
        update_val(wb_st, i, header_name_row,basin_row,opt_par_value)
    
    
    
    met_name = {"Nash":"Nash","Relative Q":"RelQ","Relative Vol":"RelVol"}
    met_name2 = ["Nash","Relative Q", "Relative Vol"]
    
    ###### Update performance metrics ##################
    
    for met in list(my_perf_metric['Event name']):
        
        for met_p in list(met_name2):
    
            mtnm = met_name[met_p]+"_"+met
    
            value = my_perf_metric[my_perf_metric['Event name']==met][met_p].iloc[0]
    
            update_val(wb_st, mtnm, header_name_row,basin_row,value)
                    
            
            #my_excel.loc[my_excel['Basin Name'] == basin_name,mtnm] = my_perf_metric[my_perf_metric['Event name']==met][met_p].iloc[0]
    
    #####################################################################################################################################
    
    #### update averages #######
    
    update_val(wb_st, "Avg_Nash", header_name_row,basin_row,Nash_mean)
    update_val(wb_st, "Avg_RelQ", header_name_row,basin_row,RelQ_mean)
    update_val(wb_st, "Avg_RelVol", header_name_row,basin_row,RelVol_mean)
    
    ####### Update other values ######
    
    update_val(wb_st, "Modeler Assigned", header_name_row,basin_row,modeler_name)
    update_val(wb_st, "Calibration Folder", header_name_row,basin_row,model_dir)
    update_val(wb_st, "Flow meter", header_name_row,basin_row,flow_dir_file)
    update_val(wb_st, "Rain gauge", header_name_row,basin_row,rain_dir_file)
    
    ###### Event_detail Sheet ####################

    wb_st_2 = wb_obj['Event_detail']
    basin_name_evnt_sht = 'A'
    
    
    for cell in wb_st_2[basin_name_evnt_sht]:
        if(cell.value == basin_name): #We need to check that the cell is not empty.
            #if 'Table' in cell.value: #Check if the value of the cell contains the text 'Table'
                #print('Found cell with name: {} at row: {} and column: {}. In cell {}'.format(cell.value,cell.row,cell.column,cell))
                basin_row_s2 = cell.row
    
    event_col_strt = 2
    
    
    ####### Update Event dates ##################
    
    #### Rearrange event dataframe ##########
    adf_event_data.columns = list(event_data.columns)
    
    event_data = pd.concat([event_data, adf_event_data]).reset_index(drop = True)
    
    
    for eve in list(event_data['Event']):
        strt_dt = event_data[event_data['Event']==eve]['Start Date'].iloc[0]
        end_dt = event_data[event_data['Event']==eve]['End Date'].iloc[0]
        
        for dt in [strt_dt, end_dt]:
            celll = wb_st_2.cell(row = basin_row_s2, column = event_col_strt)
            celll.value = str(dt)
            event_col_strt = event_col_strt + 1
            
    
    ##### Save the updated file ###################
    wb_obj.save(Main_excel_file)
    
    print("Main excel file successfully updated!")
    