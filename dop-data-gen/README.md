# GPS Visibility Analysis

Author: Mark Mendiola, The Aerospace Corporation.

## Dependencies

Python 3.11.

### Install

```sh
python -m pip install -r requirements.txt
```

## Configure

### The DOP Script

Configure the following environment variables to run the `generate_dop_data.py` script.

| env var name | default | description |
| --- | --- | --- |
| `TIME_KEY` | None | Required. The timestamp used to compute the DOP in RFC3339 format. YYYY-mm-DDT-HH-MM-SSZ (e.g. 2024-08-21T15:40:30Z) |
| `SEM_ALM_FILE` | `current_sem_week.txt` | The file path to the SEM almanac file. |
| `DOP_OUTPUT_FILE` | `dop_output.txt` | The output file for the DOP metrics. |

### InfluxDB V1

Configure the following environment variables to run the `write_to_influxdb.py` script.

| env var name | default | description |
| --- | --- | --- |
| `TIME_KEY` | None | Required. The timestamp used to compute the DOP in RFC3339 format. YYYY-mm-DDT-HH-MM-SSZ (e.g. 2024-08-21T15:40:30Z) |
| `DOP_OUTPUT_FILE` | `dop_output.txt` | The data file from the DOP data generation script. |
| `INFLUXDB_WRITE_HOST_V1` | `None` | The InfluxDB host for the V1 write client. |
| `INFLUXDB_WRITE_DB` | `None` | The InfluxDB database to write for the V1 client. |
| `INFLUXDB_WRITE_USER` | `None` | The user credential for the V1 write client. |
| `INFLUXDB_WRITE_PASS` | `None` | The password credential for the V1 write client. |
| `INFLUXDB_WRITE_MEAS_V1` | `dop` | The InfluxDB measurement used to store the data. |

## Step 1 - Run the DOP Script

The `generate_dop_data.py` script contains a DOP data generation algorithm that outputs a data file which can then be read and input into InfluxDB V1.

> It takes a few minutes for the script to complete.

```sh
export TIME_KEY=2024-08-21T15:40:30Z
python generate_dop_data.py
```

By default, it outputs a data file called `dop_output.txt` that contains the new DOP data.

## Step 2 - Write the DOP Metrics to InfluxDB

The `write_to_influxdb.py` script reads the data from the  to store the DOP metrics onto InfluxDB version 1.8. 

```sh
export TIME_KEY=2024-08-21T15:40:30Z
export host INFLUXDB_WRITE_HOST_V1=localhost
export database INFLUXDB_WRITE_DB=mydopdb
export username INFLUXDB_WRITE_USER=myuser
export password INFLUXDB_WRITE_PASS=mypass
python write_to_influxdb.py
```

The website will query InfluxDB for this DOP data for this particular time step.
