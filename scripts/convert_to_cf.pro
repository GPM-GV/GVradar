	pro convert_to_cf,file
        !Quiet = 1

	radar = rsl_anyformat_to_radar(file,ERROR=error,/QUIET)
	pol_cf_file = file + '.cf'
        rsl_radar_to_cfradial,radar,pol_cf_file,catch=0,FIELDS=my_fields,/UFFIELDS

	print,'Done.'
        return
        end
