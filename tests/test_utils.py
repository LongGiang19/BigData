from src.utils import day_of, norm

def test_day_of():
    assert day_of("Bruteforce-Wednesday-14-02-2018_x.parquet") == "2018-02-14"
    assert day_of("Infil2-Thursday-01-03-2018_x.parquet") == "2018-03-01"
    assert day_of("02-14-2018.csv") == "2018-02-14"

def test_norm():
    assert norm("Flow Bytes/s") == "flow_bytes_s"
