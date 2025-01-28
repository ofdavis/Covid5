frame change default 
clear all 
import delimited "/Users/owen/Downloads/Provisional_COVID-19_Deaths_by_County__and_Race_and_Hispanic_Origin_20250127.csv", clear

keep if indicator=="Distribution of population (%)"
keep fipscode fipsstate totaldeaths covid19deaths
rename fipscode county 
rename fipsstate state 

* populations 
copy "https://www2.census.gov/programs-surveys/popest/datasets/2020-2021/counties/totals/co-est2021-alldata.csv" ///
	data/countypops2020.csv, replace
frame2 pops, replace
import delimited data/countypops2020.csv
keep state county popestimate2020
rename popestimate2020 pop
drop if county==0
replace county = 1000*state + county 
drop state
tempfile pops 
save "`pops'"
frame change default 
merge 1:1 county using "`pops'"
drop if _merge==2
drop _merge 


*---------- use ASEC data to choose which counties to keep ---------------------
frame2 asec, replace
use data/generated/asec_data
keep if year==2019
collapse (sum) asecwt, by(state county) 
drop if county==0
tempfile asec
save "`asec'"
frame change default 
merge 1:1 county using "`asec'", gen(_merge_asec)
drop if _merge==2 // two counties, changes? 

* county2 defn: county or state 
gen county2 = county if _merge_asec==3
replace county2 = state*100000 if county2==. 
drop _merge_asec


* ------------------------------------ collapse----------------------------------
collapse (sum) covid19deaths totaldeaths pop (first) state, by(county2)
gen covidrate = covid19deaths/pop
gen deathrate = totaldeaths/pop

order state county2 covidrate deathrate
sort state county2

* state-level counties back to 0 
replace county2=0 if county2>=100000
rename county2 county 

keep state county *rate 

* save 
compress 
save data/generated/covid, replace
