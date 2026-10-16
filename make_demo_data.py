"""Create an independent synthetic fixture; this is not the historical dataset."""
import csv
import math
import random
from pathlib import Path


def generate(path="dataset.csv", rows=240):
    rng = random.Random(17)
    features = ["tempmax", "tempmin", "temp", "feelslikemax", "feelslikemin", "feelslike", "pm2.5", "no2", "co2", "dew", "humidity", "precip", "precipprob", "precipcover", "windgust", "windspeed", "winddir", "pressure", "cloudcover", "visibility", "solarradiation", "solarenergy", "uvindex", "severerisk", "moonphase", "tempRange", "heatIndex", "severityScore"]
    fields = ["datetimeEpoch", *features, "sunriseEpoch", "sunsetEpoch", "month", "dayOfWeek", "isWeekend", "healthRiskScore"]
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}; choose another path.")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i in range(rows):
            record = {name: round(rng.uniform(0, 100), 4) for name in features}
            epoch = 1704067200 + i * 86400
            record.update(datetimeEpoch=epoch, sunriseEpoch=epoch+21600, sunsetEpoch=epoch+64800, month=(i//30)%12+1, dayOfWeek=i%7, isWeekend=int(i%7>=5))
            # Arbitrary demonstration target, not a health model or real observations.
            record["healthRiskScore"] = round(.02*record["pm2.5"] + .01*record["no2"] + math.sin(i/20) + rng.gauss(0, .15), 4)
            writer.writerow(record)
    return path


if __name__ == "__main__":
    print(f"Wrote synthetic demo data to {generate()}. Historical metrics do not apply.")
