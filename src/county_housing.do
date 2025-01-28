* data setup 
clear all 
cd "/users/owen/Covid5"

* download and import data
* copy "https://www.fhfa.gov/hpi/download/annually/hpi_at_bdl_county.xlsx" "data/fhfa.xlsx", replace
import excel data/fhfa, cellrange(A7) firstrow clear

* variable case and destring, drops  
rename *, lower
foreach var in fipscode year hpi { 
	destring `var', replace
}
rename county countyname
rename fipscode county
drop hpiw* annualchange

* panel setup 
xtset county year

*---------------------------------- Fix NYC ------------------------------------
* bring in CBSA data to replace NY county with due to missing data 
frame2 cbsa, replace 
* copy "https://www.fhfa.gov/hpi/download/annually/hpi_at_bdl_cbsa.xlsx" "data/fhfa_cbsa.xlsx", replace
import excel data/fhfa_cbsa, cellrange(A7) firstrow
rename *, lower 
keep cbsa hpi year 
destring hpi, replace force 
destring year,replace
destring cbsa, replace 

keep if cbsa==35620
gen county = 36061
rename hpi hpi_cbsa

tempfile cbsa 
save "`cbsa'"
frame change default 
merge 1:1 year county using "`cbsa'"
drop if _merge==2
drop _merge
xtset county year

* Manhattan: fill in 2016 and 2018-2019 by interpolation 
replace hpi = (l.hpi + f.hpi)/2 if county==36061 & year==2016
replace hpi = l.hpi + (f2.hpi-l.hpi)/3 if county==36061 & year==2018
replace hpi = l2.hpi + 2*(f.hpi-l2.hpi)/3 if county==36061 & year==2019

* Manhattan: apply annual cbsa growth to post-2021
gen growth_cbsa = (hpi_cbsa-l.hpi_cbsa)/l.hpi_cbsa
replace hpi = l.hpi*(1+growth_cbsa) if county==36061 & year==2022
replace hpi = l.hpi*(1+growth_cbsa) if county==36061 & year==2023

drop growth_cbsa hpi_cbsa cbsa

*-------------------------- reduce to 2019-2023 growth -------------------------
* apply st-level growth to those missing 2019 and 2023 (+ prior years for multi-yr miss)
bys state year: egen hpi_st = mean(hpi)
xtset county year
gen growth_st = (hpi_st-l.hpi_st)/l.hpi_st 
replace hpi = l.hpi*(1+growth_st) if year==2018 & hpi==.
replace hpi = l.hpi*(1+growth_st) if year==2019 & hpi==.
replace hpi = l.hpi*(1+growth_st) if year==2021 & hpi==. // this is just hickman, ky 
replace hpi = l.hpi*(1+growth_st) if year==2022 & hpi==.
replace hpi = l.hpi*(1+growth_st) if year==2023 & hpi==.

keep if inlist(year,2019,2023)
gen dp = (hpi-l4.hpi)/l4.hpi
keep if year==2023 
keep state countyname county dp year


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

* fix up state name/fips
rename state state_abbrev
statastates, a(state_abbrev)
label values state_fips statefip_lbl
drop state_abbrev statefip state_name _merge
rename state_fips state

* county2 defn: county or state 
gen county2 = county if _merge_asec==3
replace county2 = state*100000 if county2==. 

* fix names 
gen countyname2 = countyname if _merge_asec==3
decode state, gen(statename)
replace countyname2=statename if _merge_asec<3
labmask county2, values(countyname2)
drop countyname2 statename 

drop _merge_asec




* *-------------------------- county pops for collapse -------------------------- 
copy "https://www2.census.gov/programs-surveys/popest/datasets/2020-2023/counties/totals/co-est2023-alldata.csv" ///
	data/countypops.csv, replace
frame2 pops, replace
import delimited data/countypops.csv
keep state county popestimate2023
rename popestimate2023 pop
drop if county==0
replace county = 1000*state + county 
drop state
tempfile pops 
save "`pops'"
frame change default 
merge 1:1 county using "`pops'"

drop if _merge==2 
replace pop=1 if _merge==1 // for CT counties -- by inspection, dp is fairly even across CT 
drop _merge 

* ------------------------------------ collapse----------------------------------
collapse (mean) dp (first) state [fw=pop], by(county2)
label values state statefip_lbl
order state county2 dp
sort state county2

* state-level counties back to 0 
replace county2=0 if county2>=100000
rename county2 county 

* save 
compress 
save data/generated/housing, replace






