clear all 
set maxvar 23000
global path "/Users/owen/Covid5/"
cd "${path}/data/HRS/raw/"

* create temp folder 
cap mkdir ${path}/data/HRS/raw/temp // might already exist 

* ------------------------- loop through years ---------------------------------
foreach yr in 16 18 20 22 {

	* copy core file to temp folder 
	! rm -rf ${path}/data/HRS/raw/temp 
	! mkdir ${path}/data/HRS/raw/temp
	cd "${path}/data/HRS/raw/"x
	copy "h`yr'core.zip" "temp/h`yr'core.zip", replace

	* unzip core file 
	cd "${path}/data/HRS/raw/temp"
	unzipfile "h`yr'core.zip"

	* clean up extraneous stuff 
	foreach file in "h`yr'cb.zip" "h`yr'core.zip" ///
					"h`yr'qn.zip" "h`yr'sas.zip" "h`yr'sps.zip" {
		erase `file'
	}
	
	* move pdf into pdf folder 
	cd "${path}/data/HRS/raw"
	copy "temp/h`yr'dd.pdf" "pdf/h`yr'dd.pdf", replace 
	cd "${path}/data/HRS/raw/temp"
	erase "h`yr'dd.pdf"

	* unzip da, sta file 
	unzipfile "h`yr'da.zip"
	unzipfile "h`yr'sta.zip"
	erase "h`yr'da.zip"
	erase "h`yr'sta.zip"

	* ------ loop through desired sections ------
	local covid "" 
	if `yr'==20 local covid "COV_R"
	local M1 "M1_R" 
	if `yr'==22 local M1 "M_R"
	local M2 "M2_R" 
	if `yr'==22 local M2 ""
	
	foreach sect in "PR_MC" "A_R" "A_H" "B_R" "C_R" /*"E_H"*/ "H_H" "J_R" "J3_R" /*`M1' `M2' "P_R" "Q_H"*/ `covid' {

		cd "${path}/data/HRS/raw/temp"
		* delete the "c:\hrsYYYY\data\" in the filepath 
		! sed -i '' 's|c:\\hrs20`yr'\\data\\||g' H`yr'`sect'.dct

		infile using H`yr'`sect'.dct, clear
		rename *, lower 
		
		* save in sects 
		cd "${path}/data/HRS/raw/sects"
		save H`yr'`sect'.dta, replace
	}
}

*  --------------------------- get tracker ---------------------------
cd "${path}/data/HRS/raw/"
copy "trk2022v1.zip" "temp/trk2022v1.zip", replace
cd "${path}/data/HRS/raw/temp"
unzipfile "trk2022v1.zip"
! sed -i '' 's|c:\\trk2022\\data\\||g' TRK2022TR_R.dct
infile using TRK2022TR_R.dct, clear
rename *, lower 
cd "${path}/data/HRS/raw/sects"
save tracker.dta, replace

*  --------------------------- get rand ---------------------------
cd "${path}/data/HRS/raw/"
copy "randhrs1992_2020v2_STATA.zip" "temp/randhrs1992_2020v2_STATA.zip", replace
cd "${path}/data/HRS/raw/temp"
unzipfile "randhrs1992_2020v2_STATA.zip"

* bring in, keep only waves needed (will do more variable selection later)
use randhrs1992_2020v2, clear
keep hhid pn hhidpn in* ra* h13* h14* h15* r13* r14* r15* s13* s14* s15* 
cd "${path}/data/HRS/raw/sects"
save rand.dta, replace 


* clean up temps 
cd "${path}/data/HRS/raw/"
! rm -rf ${path}/data/HRS/raw/temp 












