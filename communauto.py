import numpy as np
import pandas as pd


TAXES = 1.14975

def trip_infos(trip_km, hours_to, hours_there, weekly_freq, weekend):
	# travel distance and duration are computed for round-trip
	infos = {
		'trip_km': 2 * trip_km,
		'trip_hours': 2 * hours_to + hours_there,
		'weekly_freq': weekly_freq,
		'weekend': weekend,
		'time_to': str(int(60 * 2 * hours_to)) + ' mins',
		'time_there': str(int(hours_there)) + ' hours',
	}

	reality_mult = 1.05  # multiplier to better reflect real distance
	infos['trip_km'] = infos['trip_km'] * reality_mult

	infos['km_total'] = str(round(infos['trip_km'], 2)) + ' km'
	infos['time_total'] = str(infos['trip_hours']) + ' hours'

	return infos


LD_RATE = {'day_rate': 24.95, 'km_cost': .17, 'km_extra': .13, 'km_threshold': 300}
WW_RATE = {'day_rate': 19.20, 'km_cost': .00, 'km_extra': .43, 'km_threshold':  40}

plans = [
	['Lib Base', 'flex', 0.0,  0.00, 12.00, 50.00, 50.00, .00, .20, 100, .00, 0., False, False],
	['Lib Plus',	'all', 0.0,  3.33,  6.25, 50.00, 35.00, .16, .16, 100, .30, 3., False, False],
	['Éco Base',	'all', 500,  3.33,  3.35, 26.80, 26.80, .41, .30,  50, .30, 3.,  True, False],
	['Éco Plus',	'all', 500, 12.50,  2.95, 23.80, 23.80, .33, .24,  50, .30, 3.,  True, False],
	['Éco Extra',	'all', 500, 30.00,  2.45, 19.80, 19.80, .24, .24,  50, .30, 3.,  True,  True],
]

columns = [
	'plan_name',
	'car_type',
	'deposit',
	'monthly_cost',
	'hour_cost',
	'day_cost',
	'day_extra',
	'km_cost',
	'km_extra',
	'km_threshold',
	'wknd_hour_extra',
	'wknd_day_extra',
	'longdist_rate',
	'workweek_rate',
]

disp_cols = [
	'plan_name',
	'hour_cost',
	'day_cost',
	'km_cost',
]

trips = {
	'syl_dan': trip_infos(trip_km=62.3, hours_to=1.0, hours_there=6.0, weekly_freq=1/5, weekend=True),
	'parents': trip_infos(trip_km=40.1, hours_to=.75, hours_there=28., weekly_freq=1/5, weekend=True),
	#'chalet':  trip_infos(trip_km=149., hours_to=2.0, hours_there=40., weekly_freq=1/8, weekend=True),
	'sorties': trip_infos(trip_km=170., hours_to=2.0, hours_there=10., weekly_freq=1/8, weekend=True),
	#'mirabel': trip_infos(trip_km=68.9, hours_to=1.0, hours_there=1.5, weekly_freq=1.0, weekend=False),
	'mirabel': trip_infos(trip_km=36.5, hours_to=.67, hours_there=1.5, weekly_freq=1.0, weekend=False),
	#'stage':   trip_infos(trip_km=13.3, hours_to=.75, hours_there=6.0, weekly_freq=4.0, weekend=False),
}



plans = pd.DataFrame(data=plans, columns=columns)
print(plans[['plan_name'] + [col for col in plans.columns if 'cost' in col]])
print()


weekend_day_cost = plans['day_cost'] + plans['wknd_day_extra']
weekend_hour_cost = plans['hour_cost'] + plans['wknd_hour_extra']
plans['day_threshold'] = plans['day_cost'] / plans['hour_cost']
plans['day_threshold_wknd'] = weekend_day_cost / weekend_hour_cost


total_km, total_hours = 0, 0

for trip, infos in trips.items():
	# display trip infos
	num_tabs = 1 - len(trip) // 8
	if infos['weekly_freq'] < 1:
		print('{}{}-> every {} weeks'.format(trip, '\t' * num_tabs, 1 / infos['weekly_freq']))
	else:
		print('{}{}-> {} time(s) a weeks'.format(trip, '\t' * num_tabs, infos['weekly_freq']))

	total_km += infos['trip_km']
	total_hours += infos['trip_hours']

	# normal rate
	base_km_cost = plans['km_cost'] * np.minimum(infos['trip_km'], plans['km_threshold'])
	extra_km_cost = plans['km_extra'] * np.maximum(0, infos['trip_km'] - plans['km_threshold'])
	trip_km_cost = base_km_cost + extra_km_cost

	if infos['weekend'] is False:
		hour_tarif = plans['hour_cost']
		first_day_tarif = plans['day_cost']
		extra_days_tarif = plans['day_extra']
		day_threshold = plans['day_threshold']
	else:
		hour_tarif = plans['hour_cost'] + plans['wknd_hour_extra']
		first_day_tarif = plans['day_cost'] + plans['wknd_day_extra']
		extra_days_tarif = plans['day_extra'] + plans['wknd_day_extra']
		day_threshold = plans['day_threshold_wknd']


	trip_hours = infos['trip_hours'] % 24
	trip_days = infos['trip_hours'] // 24

	use_hour_mask = np.asarray(trip_hours < day_threshold)
	num_hours = trip_hours * use_hour_mask
	num_days = trip_days + ~use_hour_mask

	hours_cost = num_hours * hour_tarif
	first_day_cost = first_day_tarif * np.minimum(1, num_days)
	extra_days_cost = extra_days_tarif * np.maximum(0, num_days - 1)
	trip_duration_cost = hours_cost + first_day_cost + extra_days_cost

	normal_rate = trip_km_cost + trip_duration_cost


	# long-distance rate
	base_km_cost = LD_RATE['km_cost'] * np.minimum(infos['trip_km'], LD_RATE['km_threshold'])
	extra_km_cost = LD_RATE['km_extra'] * np.maximum(0, infos['trip_km'] - LD_RATE['km_threshold'])
	trip_km_cost = base_km_cost + extra_km_cost

	use_hour_mask = np.asarray(trip_hours < 3)
	num_hours = trip_hours * use_hour_mask
	num_days = trip_days + ~use_hour_mask
	hours_cost = num_hours * 10
	days_cost = num_days * LD_RATE['day_rate'] + 5.0  # 5$ extra for first day
	trip_duration_cost = hours_cost + days_cost

	use_ld_rate_mask = np.asarray(plans['longdist_rate'])
	longdist_rate = (trip_km_cost + trip_duration_cost) + 1e3 * ~use_ld_rate_mask


	# workweek rate
	workweek_rate = np.asarray([np.nan] * len(plans))
	if infos['weekend'] is False:
		base_km_cost = WW_RATE['km_cost'] * np.minimum(infos['trip_km'], WW_RATE['km_threshold'])
		extra_km_cost = WW_RATE['km_extra'] * np.maximum(0, infos['trip_km'] - WW_RATE['km_threshold'])
		trip_km_cost = base_km_cost + extra_km_cost

		use_ww_rate_mask = np.asarray(plans['workweek_rate'])
		workweek_rate = (WW_RATE['day_rate'] + trip_km_cost) + 1e3 * ~use_ld_rate_mask

	# get best rate for the trip
	rate_prices = np.vstack([normal_rate, longdist_rate, workweek_rate]).T
	trip_cost = np.nanmin(rate_prices, axis=1)

	print(np.round(rate_prices, 2))
	print()

	plans[trip] = TAXES * trip_cost * infos['weekly_freq'] * 52 / 12





print()

trips = pd.DataFrame(data=trips)
print(trips[3:].T)
print()



print('Monthly distance: {}km'.format(round(total_km, 2)))
print('Monthly total time: {} hours'.format(round(total_hours, 1)))
print()

cost_columns = list(trips.keys())
plans['monthly_cost'] += plans[cost_columns].sum(axis=1)
plans['monthly_cost'] *= TAXES
cost_columns.extend(['monthly_cost'])

plans[cost_columns] = plans[cost_columns].round(2)
plans.sort_values(by='monthly_cost', ascending=True, inplace=True)
print(plans[columns[:3] + cost_columns])
print()

