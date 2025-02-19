* data setup 
clear all 
cd "/users/owen/Covid5"
global styear 2000

/* -----------------------------------------------------------------------------
									get IPUMS CPS data 
 -----------------------------------------------------------------------------
global asec 0
global bms 1
global start_yr = $styear
global start_mo = 1
global end_yr = 2024
global end_mo = 12
global var_list = "YEAR SERIAL MONTH MISH COUNTY STATEFIP METRO PERNUM WTFINL CPSIDP AGE SEX RACE MARST HISPAN NATIVITY EMPSTAT LABFORCE OCC2010 IND1990 CLASSWKR UHRSWORK1 AHRSWORK1 WKSTAT ABSENT EDUC EARNWT LNKFW1YWT LNKFW1MWT HOURWAGE2 EARNWEEK2 DIFFANY DIFFHEAR DIFFEYE DIFFREM DIFFPHYS DIFFMOB DIFFCARE VETSTAT SPLOC HRHHID FAMSIZE NCHILD ELDCH YNGCH WHYUNEMP DURUNEMP"

do "/Applications/Stata/ado/personal/ipums_get.do"

save "data/cps_raw.dta", replace
*/

/* -----------------------------------------------------------------------------
								make date var, other basics 
 ----------------------------------------------------------------------------- */
use data/cps_raw.dta, clear
cap drop hwtfinl asecflag  
format cpsidp %15.0f
gen mo = ym(year,month) 
format mo %tm

* covid dummy
gen covid = mo>=`=tm(2020m4)'

/* -----------------------------------------------------------------------------
									Backcast
This code implements the CPS weight backcasting described here: https://github.com/TheHamiltonProjectResearch/CPS-Population-Adjustment-Backcast-2012-22
The adjusted weights will be used for everything else 
-----------------------------------------------------------------------------*/

* Race/Ethnicity
gen ethnic=.
replace ethnic=6 if inrange(hispan,100,902) & ethnic==. 
replace ethnic=1 if race==100  & ethnic==. 
replace ethnic=2 if race==200  & ethnic==. 
replace ethnic=4 if inlist(race, 650,651) & ethnic==. // for 00-02, PI is in AAPI 
replace ethnic=5 if race==300 & ethnic==. 
replace ethnic=5 if inrange(race, 652, 830) & ethnic==. 
label define ethniclab 1 "White" 2 "Black" 4 "Asian" 5 "Other" 6 "Hispanic" , replace
label values ethnic ethniclab

* Age Groups
gen agegr=.
replace agegr = 1 if inrange(age, 16, 17)
replace agegr = 2 if inrange(age, 18, 19) 
replace agegr = 3 if inrange(age, 20, 24) 
replace agegr = 4 if inrange(age, 25, 54) 
replace agegr = 5 if inrange(age, 35, 44) 
replace agegr = 6 if inrange(age, 45, 54) 
replace agegr = 7 if inrange(age, 55, 64) 
replace agegr = 8 if inrange(age, 65, 90) 

label define agegrlab 1 "16-17" 2 "18-19" 3 "20-24" 4 "25-34" 5 "35-44" 6 "45-54" 7 "55-64" 8 "65+"
label values agegr agegrlab

*merge backcast file in
merge m:1 month year sex agegr ethnic using data/cps_backcast, keepusing(ratio_adj) 
drop if _merge==2
drop _merge 

replace ratio_adj=1 if ratio_adj==.

foreach wt in wtfinl earnwt lnkfw1mwt lnkfw1ywt { 
	replace `wt' = `wt' * ratio_adj 
}

drop ratio_adj agegr

replace wtfinl = round(wtfinl)
replace earnwt = round(earnwt)
replace lnkfw1mwt = round(lnkfw1mwt)
replace lnkfw1ywt = round(lnkfw1ywt)


/*----------------------------------------------------------------------------
								 Spouses
  ----------------------------------------------------------------------------*/
* Marital status 
gen married = 0
replace married = 1 if marst==1
label variable married "Marital status"
label define married 0  "Unmarried" 1 "Married" 
label values married married

* Create new frame to make and merge back in spouse data 
frame copy default spouse, replace
frame change spouse
	keep sploc marst serial cpsid hrhhid age year month
	keep if marst==1 & sploc!=0

	rename age age_sp
	rename sploc pernum
	rename cpsid cpsid_sp // for testing 

* merge back in
frame change default 
frame spouse: tempfile sp 
frame spouse: save "`sp'"
merge 1:1 serial pernum year month hrhhid using "`sp'", nogen
frame drop spouse

* Age groups, spouse. Note: 120 respondents 55+  have spouse age<18
	gen agegrp_sp = .
replace agegrp_sp = 55 if inrange(age_sp, 15, 59)
replace agegrp_sp = 60 if inrange(age_sp, 60, 61)
replace agegrp_sp = 62 if inrange(age_sp, 62, 64)
replace agegrp_sp = 65 if inrange(age_sp, 65, 69)
replace agegrp_sp = 70 if inrange(age_sp, 70, 90)

* for married respondents who have missing age_sp, make agegrp_sp same as agegrp
misstable sum agegrp_sp if married==1 
replace agegrp_sp = 55 if inrange(age, 15, 59) & married==1 & agegrp_sp==.
replace agegrp_sp = 60 if inrange(age, 60, 61) & married==1 & agegrp_sp==.
replace agegrp_sp = 62 if inrange(age, 62, 64) & married==1 & agegrp_sp==.
replace agegrp_sp = 65 if inrange(age, 65, 69) & married==1 & agegrp_sp==.
replace agegrp_sp = 70 if inrange(age, 70, 90) & married==1 & agegrp_sp==.

* create a separate category agegrp_sp=0 for these
replace agegrp_sp = 0 if married==0

* Drop younger observations -- no longer needed for spousal match 
drop if age<50

* family size 
gen child_any = nchild!=0
gen child_yng = yngch<=18
gen child_adt = eldch>=18 & eldch<99

label variable child_any "Own child in house"
label variable child_yng "Own young child in house"
label variable child_adt "Own adult child in house"
label define yesno 0 "No" 1 "Yes"
label values child_any yesno 
label values child_yng yesno 
label values child_adt yesno 

tab yngch if nchild>1

drop if empstat==0 // one weird obs in 2002 with most missing 

/*----------------------------------------------------------------------------
								 Demographic vars 
  ----------------------------------------------------------------------------*/
replace sex = sex-1
label define sex 0 "Men" 1 "Women"
label values sex sex 

* age: make consistent (2002-2004 will not have any 85+)
replace age = 85 if age>=85
replace age = 80 if inrange(age,80,84)

gen agesq = age^2
gen agecub = age^3

* educ
gen educ_ = 0 
replace educ_ = 1 if educ==73
replace educ_ = 2 if inrange(educ,81,110)
replace educ_ = 3 if educ==111
replace educ_ = 4 if inrange(educ,112,125)

label define educ 0 "Less than high school" 1 "High school degree" 2 "Some college" 3 "Bachelor's degree" 4 "Advanced degree", replace 
label values educ_ educ

drop educ
rename educ_ educ
label variable educ "Education"

* race
drop race 
rename ethnic race
label variable race "Race/ethnicity"

* native 
tab nativity
recode nativity (0=3) (1=0) (2/4=1) (5=2)
label define nativity 0 "Native-born, both parents" 1 "Native-born, foreign parent(s)" 2 "Foreign-born" 3 "Unknown"
label values nativity nativity

gen foreign = inrange(nativity,2,3)
label variable foreign "Nationality"
label define foreign 0 "Native-born" 1 "Foreign-born", replace
label values foreign foreign

* metro area
rename metro metro_cps
gen metro = inrange(metro_cps, 2, 4)
label variable metro "Metro status"
label define metro 0 "Rural" 1 "Metro"
label values metro metro 

* veteran 
tab vetstat
replace vetstat=0 if vetstat==1
replace vetstat=1 if vetstat==2
rename vetstat vet
label drop vetstat_lbl 
label define vet 0 "Not a veteran" 1 "Veteran"
label values vet vet 

* disability 
gen disable = 0 
replace disable=1 if diffany==2

replace diffrem = diffrem==2
replace diffphys = diffphys==2
replace diffmob = diffmob==2

* some labeling
label define mish 1 "MIS 1" 2 "MIS 2" 3 "MIS 3" 4 "MIS 4" 5 "MIS 5" 6 "MIS 6" 7 "MIS 7" 8 "MIS 8"
label values mish mish 

label define covid_lbl 0 " " 1 "Covid"
label values covid covid_lbl

label var mo "date"

drop *_cps 


/*----------------------------------------------------------------------------
								Employment vars 
  ----------------------------------------------------------------------------*/
*employment status
gen emp=.
replace emp=1 if empstat==10 | empstat==12 // employed
replace emp=2 if empstat==21 | empstat==22 // unemployed
replace emp=3 if empstat==32 | empstat==34 					// not in LF other
replace emp=4 if empstat==36 // not in LF ret
drop if emp==. // armed forces

*retired status, employed status
gen retired = empstat==36
gen employed = emp==1
gen unem = emp==2
gen nlf = emp==3
gen unable = empstat==32
gen nlf_oth = empstat==34 
gen untemp = whyunemp==1
gen unlose = whyunemp==2

* self employed
gen self = inrange(classwkr, 10, 14)

* public vs private
gen govt = inrange(classwkr, 24, 28)

* full time worker
gen ft = (wkstat>=10 & wkstat<=15) | wkstat==50

* absent 
gen absnt = absent==3 

* duration unemployment 
gen dur = durunemp if unem==1 
assert dur<999 if unem==1

/*----------------------------------------------------------------------------
								Earnings vars 
------------------------------------------------------------------------------*/
* tag topcodes 
bys mo mish: egen hourtop__ = max(hourwage2) if hourwage2<999
bys mo mish: egen hourtop_ = max(hourtop__) 
bys mo mish: egen weektop__ = max(earnweek2) if hourwage2<9999
bys mo mish: egen weektop_ = max(weektop__) 
gen hourtop = hourwage2==hourtop_
gen weektop = earnweek2==weektop_

* indicate if only weekly earn vars avail 
gen week_only = 1 if earnweek2<9999 & hourwage2>999

* create wage var 
gen wage = hourwage2 if hourwage2<999
replace wage = earnweek2/uhrswork1 if week_only==1 & uhrswork1<997
replace wage = earnweek2/ahrswork1 if week_only==1 & uhrswork1==997

* indicate if implied hourly>hourtop 
gen wageflag = wage>hourtop_ & wage<. & week_only==1

* get cpi 
frame2 cpi, replace 
import fred CPIAUCSL, daterange(${styear}-01-01 2024-12-01) aggregate(monthly)
gen mo = ym(year(daten), month(daten))
format mo %tm
gen cpi = CPIAUCSL/CPIAUCSL[1]
keep mo cpi 
tempfile cpi 
save "`cpi'"
frame change default 
merge m:1 mo using "`cpi'"
assert _merge!=1
drop if _merge==2
drop _merge

* deflate hourwage 
replace wage = wage/cpi if wage<.

* create wage quartiles 
xtile earn_qtr = wage, n(4) 


* clean up 
drop hourtop_* weektop_* week_only cpi




/*----------------------------------------------------------------------------
							industry and occupation 
  ---------------------------------------------------------------------------- */
*maj industry grps
gen ind_maj = .
replace ind_maj = 1 if  inrange(ind1990, 10, 32)    	// ag etc
replace ind_maj = 2 if  inrange(ind1990, 40, 50)    	// mining
replace ind_maj = 3 if  inrange(ind1990, 60, 60)    	// construction
replace ind_maj = 4 if  inrange(ind1990, 100, 392)   	// manuf
replace ind_maj = 5 if  inrange(ind1990, 400, 472) 		// trans/util
replace ind_maj = 6 if  inrange(ind1990, 500, 571)   	// wholesale
replace ind_maj = 7 if  inrange(ind1990, 580, 691)   	// retail
replace ind_maj = 8 if  inrange(ind1990, 700, 712)   	// financial
replace ind_maj = 9 if  inrange(ind1990, 721, 760)   	// biz and repair serv
replace ind_maj = 10 if inrange(ind1990, 761, 791)   	// personal serv
replace ind_maj = 11 if inrange(ind1990, 800, 810)   	// entertainment and rec
replace ind_maj = 12 if inrange(ind1990, 812, 893)   	// prof and related
replace ind_maj = 13 if inrange(ind1990, 900, 932)   	// public admin
replace ind_maj = 14 if inrange(ind1990, 940, 998)   	// military

label define ind_maj_lbl 1 "Agriculture and related" ///
						   2 "Mining" ///
						   3 "Construction" ///
						   4 "Manufacturing" ///
						   5 "Transportation and utilities" /// 
						   6 "Wholesale trade" ///
						   7 "Retail trade" ///
						   8 "Financial activities" ///
						   9 "Business and repair services" /// 
						   10 "Personal services" ///
						   11 "Entertainment and recreational services" ///
						   12 "Professional and related services" ///
						   13 "Public administration"  ///
						   14 "Military" , replace
label values ind_maj ind_maj_lbl 

* major occ groups 
	gen occ_maj=.
replace occ_maj=1	if	occ2010>=10   & occ2010<=440
replace occ_maj=2	if	occ2010>=500  & occ2010<=960
replace occ_maj=3	if	occ2010>=1000 & occ2010<=1240
replace occ_maj=4	if	occ2010>=1300 & occ2010<=1560
replace occ_maj=5	if	occ2010>=1600 & occ2010<=1980
replace occ_maj=6	if	occ2010>=2000 & occ2010<=2060
replace occ_maj=7	if	occ2010>=2100 & occ2010<=2180
replace occ_maj=8	if	occ2010>=2200 & occ2010<=2555
replace occ_maj=9	if	occ2010>=2600 & occ2010<=2960
replace occ_maj=10	if	occ2010>=3000 & occ2010<=3550
replace occ_maj=11	if	occ2010>=3600 & occ2010<=3655
replace occ_maj=12	if	occ2010>=3700 & occ2010<=3960
replace occ_maj=13	if	occ2010>=4000 & occ2010<=4160
replace occ_maj=14	if	occ2010>=4200 & occ2010<=4255
replace occ_maj=15	if	occ2010>=4300 & occ2010<=4655
replace occ_maj=16	if	occ2010>=4700 & occ2010<=4965
replace occ_maj=17	if	occ2010>=5000 & occ2010<=5940
replace occ_maj=18	if	occ2010>=6005 & occ2010<=6130
replace occ_maj=19	if	occ2010>=6200 & occ2010<=6950
replace occ_maj=20	if	occ2010>=7000 & occ2010<=7640
replace occ_maj=21	if	occ2010>=7700 & occ2010<=8990
replace occ_maj=22	if	occ2010>=9000 & occ2010<=9760
replace occ_maj=23	if	occ2010>=9830 & occ2010<9999

// value labels
label define occ_maj_lbl 1 "Management " ///
						 2 "Business and Financial Operations " ///
						 3 "Computer and Mathematical " ///
						 4 "Architecture and Engineering " ///
						 5 "Life, Physical, and Social Science " ///
						 6 "Community and Social Services " ///
						 7 "Legal " ///
						 8 "Education, Training, and Library " ///
						 9 "Arts, Design, Entertainment, Sports, and Media " ///
						 10 "Healthcare Practitioners and Technical " ///
						 11 "Healthcare Support " ///
						 12 "Protective Service " ///
						 13 "Food Preparation and Serving Related " ///
						 14 "Building and Grounds Cleaning and Maintenance " ///
						 15 "Personal Care and Service " ///
						 16 "Sales and Related " ///
						 17 "Office and Administrative Support " ///
						 18 "Farming, Fishing, and Forestry " ///
						 19 "Construction Trades and Extraction Workers" ///
						 20 "Installation, Maintenance, and Repair Workers" ///
						 21 "Production " ///
						 22 "Transportation and Material Moving " ///
						 23 "Armed Forces"
label values occ_maj occ_maj_lbl

rename occ2010 occ 
rename ind1990 ind


*-------------------------- ind-occ avg wages ----------------------------------
egen ind_occ = group(ind occ)
bys ind_occ: gen io_n = _N
*bys ind_maj occ: gen imo_n = _N
*bys ind occ_maj: gen iom_n = _N
egen tagio = tag(ind occ) 
*egen tagimo = tag(ind_maj occ) 
*egen tagiom = tag(ind occ_maj) 
gsort -io_n
*br ind_maj occ_maj ind_occ io_n imo_n iom_n if tagio==1


*windorize wage 
bys ind_occ: egen wage_io_p01 = pctile(wage), p(1)
bys ind_occ: egen wage_io_p99 = pctile(wage), p(99)
gen wage_w = wage 
replace wage_w = wage_io_p01 if wage_w<wage_io_p01
replace wage_w = wage_io_p99 if wage_w>wage_io_p99 & wage_w<.

* average winsorized wage 
bys ind_occ: egen wage_io = mean(wage) 

* reg wage prediction 
reg wage i.ind i.occ if wage<. & occ<9999 & ind!=0
predict wage_io_p 
gen diff = wage_io-wage_io_p
gsort -diff
twoway scatter wage_io wage_io_p if tagio==1 & io_n>50

replace wage_io = wage_io_p if io_n<150
gen lwage_io = log(wage_io)
sum wage_io, d 

/*----------------------------------------------------------------------------
								Social security PIA 
  ----------------------------------------------------------------------------*/
* 1. For each age/year combo, provide 12 possible birth dates (year and month, assume born on last day of month)
forvalues m=1/12 { 
	gen bdate`m' = mo-((age+1)*12)+`m'-1
	format bdate`m' %tm
}

/* need a new approach here that avoids missings 
-- for each bmonth, calculate all possible PIA values  
-- rescale so that max = 1 
-- for any age<62, replace pia with pia at 62m1 
-- for any age>70, replace pia = 1 
-- create dummy indicating age in [62,69]
*/ 

frame2 pia, replace 
set obs 732 // enough for bdates between 1924-1975 
gen bdate = _n -1 + `=tm(1914m1)'
format bdate %tm 

* 2. For each birth date, calculate time until/since full retirement age
gen rdate = bdate ///
	+ (year(dofm(bdate))<=1937) * (65*12 + 0 ) ///
	+ (year(dofm(bdate))==1938) * (65*12 + 2 ) /// 
	+ (year(dofm(bdate))==1939) * (65*12 + 4 ) /// 
	+ (year(dofm(bdate))==1940) * (65*12 + 6 ) /// 
	+ (year(dofm(bdate))==1941) * (65*12 + 8 ) /// 
	+ (year(dofm(bdate))==1942) * (65*12 + 10) ///
	+ (year(dofm(bdate))==1943) * (66*12 + 0 ) ///
	+ (year(dofm(bdate))==1944) * (66*12 + 0 ) ///
	+ (year(dofm(bdate))==1945) * (66*12 + 0 ) ///
	+ (year(dofm(bdate))==1946) * (66*12 + 0 ) ///
	+ (year(dofm(bdate))==1947) * (66*12 + 0 ) ///
	+ (year(dofm(bdate))==1948) * (66*12 + 0 ) ///
	+ (year(dofm(bdate))==1949) * (66*12 + 0 ) ///
	+ (year(dofm(bdate))==1950) * (66*12 + 0 ) ///
	+ (year(dofm(bdate))==1951) * (66*12 + 0 ) ///
	+ (year(dofm(bdate))==1952) * (66*12 + 0 ) ///
	+ (year(dofm(bdate))==1953) * (66*12 + 0 ) ///
	+ (year(dofm(bdate))==1954) * (66*12 + 0 ) ///
	+ (year(dofm(bdate))==1955) * (66*12 + 2 ) /// 
	+ (year(dofm(bdate))==1956) * (66*12 + 4 ) /// 
	+ (year(dofm(bdate))==1957) * (66*12 + 6 ) /// 
	+ (year(dofm(bdate))==1958) * (66*12 + 8 ) /// 
	+ (year(dofm(bdate))==1959) * (66*12 + 10) ///
	+ (year(dofm(bdate))>=1960) * (67*12 + 0 )
format rdate %tm


* 3. For each possible birth date, calculate PIA 
* calculate gains 
gen gain = .
replace gain = 0.03  if inrange(year(dofm(bdate)),1914, 1924) 
replace gain = 0.035 if inrange(year(dofm(bdate)),1925, 1926) 
replace gain = 0.04  if inrange(year(dofm(bdate)),1927, 1928) 
replace gain = 0.045 if inrange(year(dofm(bdate)),1929, 1930) 
replace gain = 0.05  if inrange(year(dofm(bdate)),1931, 1932) 
replace gain = 0.055 if inrange(year(dofm(bdate)),1933, 1934) 
replace gain = 0.06  if inrange(year(dofm(bdate)),1935, 1936) 
replace gain = 0.065 if inrange(year(dofm(bdate)),1937, 1938) 
replace gain = 0.07  if inrange(year(dofm(bdate)),1939, 1940) 
replace gain = 0.075 if inrange(year(dofm(bdate)),1941, 1942) 
replace gain = 0.08  if year(dofm(bdate))>=1943

* expand to length of sample -- long enough to get FRA for every possible bdate 
qui sum rdate 
scalar min = r(min)
scalar max = r(max)
local span = max + 36 - min + 1 
di `span'
expand `span'
bys bdate: gen mo = _n
replace mo = mo - 1 + min 
format mo %tm

gen age = floor((mo-bdate)/12)

* calculate PIA 
gen pia = . 
replace pia = (-5/9)*0.01*(rdate-mo) if inrange(rdate-mo,1,36) & age>=62 & mo<rdate
replace pia = (-5/9)*0.01*36 + (-5/12)*0.01*(rdate-mo-36) if (rdate-mo)>36 & age>=62 & mo<rdate
replace pia = gain*(mo-rdate)/12 if mo>=rdate & ///
	(age<70 | (age==70 & month(dofm(bdate))==month(dofm(mo)))) 
	
* pia as factor 
gen pia1 = pia + 1 
bys bdate (mo): egen pia1_max = max(pia1)
bys bdate (mo): egen pia1_min = min(pia1)
replace pia1=pia1_max if age>=70
replace pia1=pia1_min if age<62

keep bdate mo pia1
rename pia1 pia 

* merge to default frame each possible bdate 
frame change default 
forvalues m=1/12 { 
	frame pia: rename bdate* bdate`m'
	frame pia: rename pia* pia`m'
	frame pia: tempfile pia 
	frame pia: save "`pia'"
	merge m:1 mo bdate`m' using "`pia'", nogen keep(match master)
}
*br year month age bdate* pia* 

egen pia = rowmean(pia1 pia2 pia3 pia4 pia5 pia6 pia7 pia8 pia9 pia10 pia11 pia12) 

drop bdate1-bdate12 
drop pia1-pia12 

gen ssa = inrange(age,62,69)

/*----------------------------------------------------------------------------
								urates 
  ----------------------------------------------------------------------------*/
frame2 urate, replace 
import fred UNRATE, daterange(${styear}-01-01 2024-12-01) aggregate(monthly) 
gen mo = ym(year(daten),month(daten))
format mo %tm
drop date* 
rename UNRATE ur 

* CBO urate 
frame2 urate2, replace 
import fred NROU, daterange(${styear}-01-01 2024-12-01) aggregate(quarterly) 
expand 3
sort daten
gen month = month(daten)
gen year = year(daten)
egen add = fill(0 1 2 0 1 2)
gen month1 = month+add
gen mo = ym(year,month1)
format mo %tm
drop date* month* year
rename NROU cbo 

frame change urate 
frame urate2: tempfile urate2 
frame urate2: save "`urate2'"
merge 1:1 mo using "`urate2'"
assert _merge!=1
drop if _merge==2
drop _merge

gen urhat = ur-cbo 
sum urhat if inrange(mo,`=tm(2019m10)', `=tm(2019m12)')
replace urhat = r(mean) if mo>=`=tm(2020m1)'
keep mo ur urhat

frame change default 
frame urate: tempfile urate 
frame urate: save "`urate'"
merge m:1 mo using "`urate'"
assert _merge!=1
drop if _merge==2
drop _merge

assert ur!=.

/*----------------------------------------------------------------------------
								linking
----------------------------------------------------------------------------*/
* linking 
xtset cpsidp mo 
foreach var in employed retired unem nlf mo covid {
	gen f12_`var' = f12.`var'
}
format f12_mo %tm

gen wtf12 = lnkfw1ywt 

/*----------------------------------------------------------------------------
								finalize and save 
  ----------------------------------------------------------------------------*/
compress

global basic_vars year mo month mish covid county statefip wtfinl wtf12 cpsidp age sex ///
				  vet diffrem diffphys diffmob race nativity foreign ///
				  famsize child_any child_yng child_adt agegrp_sp married ///
				  agesq agecub educ metro emp employed retired unem nlf ///
				  dur untemp unlose unable nlf_oth pia ur urhat ssa 

global work_vars self govt ft absnt hourtop weektop wage wage_io lwage_io wageflag ind_maj occ_maj 

global long_vars f12_employed f12_retired f12_mo f12_covid f12_unem f12_nlf

* long data from 2010 
preserve 
keep $basic_vars $work_vars $long_vars 
*keep if year>=2010
save data/generated/cps_data.dta, replace
restore 

/* cross-sect data from 2000
preserve 
keep $basic_vars
save data/covid_data.dta, replace
restore */ 


/* one obs per cpsidp 
use data/covid_long.dta, clear 
unique cpsidp
gen rand = runiform()
bys cpsidp: egen rank = rank(rand)
sort cpsidp mo
br cpsidp rand rank
keep if rank==1
unique cpsidp
save data/covid_long_reduced.dta, replace
*/
