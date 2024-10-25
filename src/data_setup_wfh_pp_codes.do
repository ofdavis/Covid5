*************************************************************
******* Replace occ codes to match Mongey et al codes *******
*************************************************************
* Because Mongey et al use occupation codes prior to redesign, they do not align with IPUMs data.
* The revisions of the codes below recategorize certain new occupations to related occupations that are in Mongey et al.
* None of these changes alter any of the major occupational codings used elsewhere.

* make copy of occ2010 for change/merge
gen occ_s = occ2010 

* 
replace occ_s = 40 if occ_s==30 				/* mgrs in mkting, ads, PR to ads mgrs. occ_s code 30 is split up into 3
												   types of mgrs in Mongey, but makes no diff */
replace occ_s = 136 if occ_s==130 				// HR managers to HR managers
replace occ_s = 530 if occ_s==510 | occ_s==520 	/* buyers (farm and wholesale/retail) to purchasing agents except farm/wholesale
												   Note on above: code 530 in Mongey et al is LT unem...error I think. 
												   Everywhere else it is purchasing agents and buyers. Assuming it's this. */
replace occ_s = 565 if occ_s==560 				// compliance officers x2
replace occ_s = 630 if occ_s==620 				// HR workers x2
replace occ_s = 725 if occ_s==720 				// meeting planners x2
replace occ_s = 735 if occ_s==730 				// other biz oper and mgmt specialists to mkt research analysts
replace occ_s = 1030 if occ_s==1000 			// comp sci/webdev/etc to web dev
replace occ_s = 1105 if occ_s==1100 			// network/sys admins x2
replace occ_s = 1210 if occ_s==1240 			// math sci occs to mathematicians
replace occ_s = 1965 if occ_s==1960 			// life,phys,soc sci techs x2
replace occ_s = 2025 if occ_s==2020 			// comm serv spec x2
replace occ_s = 2040 if occ_s==2060 			// misc religious workers to clergy
replace occ_s = 2145 if occ_s==2140 			// paralegals x2
replace occ_s = 2160 if occ_s==2150 			// legal support x2
replace occ_s = 2740 if occ_s==2760 			// mis sports and performers to dancers [note: musicians is high wfh for Mongey et al; oh well]
replace occ_s = 3255 if occ_s==3130 			// RNs x2
replace occ_s = 3160 if occ_s==3240 			// assorted therapists to occ therapists...not great but it's best we've got
replace occ_s = 3420 if occ_s==3410 			// health practitioner support x2
replace occ_s = 3535 if occ_s==3530 			// health technicians x2
replace occ_s = 3655 if occ_s==3650 			// misc health support workers x2
replace occ_s = 3710 if occ_s==3730 			// misc protective serv worker supervisors to first-line supervisors of police
replace occ_s = 3955 if occ_s==3950 			// Law enforcement workers, nec to all other prot serv wkrs
replace occ_s = 4610 if occ_s==4650 			// other pers care wkrs to pers care aides (latter is lwfh and hpp)
replace occ_s = 4720 if occ_s==4965 			// misc sales and related to cashiers (LWFH HPP) -- note the HWFH of other sales categories in Mongey...hmmm...
replace occ_s = 5150 if occ_s==5165 			// misc financial clerks to procurement clerks (similar edu) 
replace occ_s = 5020 if occ_s==5030 			// comms equip opers to telephone opers 
replace occ_s = 5410 if occ_s==5420 			// info and record clerks other to reservation and trns ticket agents (edu similar) 
replace occ_s = 5910 if occ_s==5940 			// office and admin support, nec to prrofreaders and copy markers (similarish edu)
replace occ_s = 6050 if occ_s==6100 			// fishing and hunting to misc agricultural 
replace occ_s = 7850 if occ_s==7855 			// food processing, nec to food cooking machin operators
replace occ_s = 8210 if occ_s==8220 			// metal and plastic, nec to tool grinders etc
replace occ_s = 8255 if occ_s==8230 			// bookbinders to printing press opers
replace occ_s = 8450 if occ_s==8460 			// textile apperl nec to upholsterers
replace occ_s = 8540 if occ_s==8550 			// woodworkers nec to woodworking machine setters
replace occ_s = 9120 if occ_s==9100 			// bus and amb drivers to bus drivers (lwfh hpp)
replace occ_s = 9130 if occ_s==9150 			// all other motor veh oper to driver/sales workers (lfwh LPP) 
replace occ_s = 9740 if occ_s==9750 			// material movers nec to refuse and recycling collectors

merge m:1 occ_s using data/wfh_pp.dta
drop if _merge==2

drop _merge
drop occ_s
