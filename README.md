# README

- Hacer un flow utilizando prefect.
    - Uno para estadísticos descriptivos
    - Otro para generar el dataset
        - Hay que detectar la fecha y quitarla
        - Quitar los saltos de línea
        - Es diferente el dataset al inicio y al final (el formato de fecha
        y distintos cambios). Intentar detectar automático.

Interesting models from huggingface:

- Parece el único de fiar haciendo pruebas, y lo bastante rápido:
https://huggingface.co/philschmid/bart-large-cnn-samsum?text=It%27s+not+exciting%2C+but+it%27s+the+kind+of+stuff+that+if+you+look+at+what+people+were+doing+before+was+dreadful%2C+right.+It%27s+like+transformative%2C+even+though+it+seems+so+boring+and+basic.+Yeah%2C%0Ayeah.+And+he+said%2C+Why+don%27t+you+much+come+work+for+me+and+this+consulting+firm%2C+the+main+the+owner+had+some+life+issues%2C+whatever%2C+and+had+to+move+on+and%2C+and+I+said%2C+Well%2C+I+don%27t+you+know%2C+I+don%27t+know+what+I%27m+doing.+He+said%2C+Oh%2C+we%27ll+figure+it+out.+There%27s+a+big+Barnes+and+Noble+at+the+top+of+the+hill+there+you+can+we+got+a+whole+section+on+it.+Yeah%2C+it+literally+Yeah.+And+and+I+did%2C+and+I+did+and+you+know%2C+it+was+I+had+a+passion+I+had+a+burning+desire%2C+right+to+learn.+It+was+fascinating+to+me.+Yeah.+And+so+I+switched+over+a+computer+science+degree+%27MIS%27+and+but+by+the+time+I+got+farther+along%2C+in+that+I+was+getting+paid+the+program+and+I+was+learning+a+ton+more+on+the+job.+So+that+kind+of+remember%0Ayour+first+experience+of+getting+paid+to+write+code.+Was+it+just+like%2C+I+can%27t+believe+actually+paying+me+to+do+this%3F+Absolutely.+Absolutely.+It+was.%0AYeah.+I+mean%2C+it%27s+literally+how+they+say%2C+you+know%2C+find+something+you+love+and+are+passionate+in+and+the+money+will+come.+And+yeah%2C+it+did.+It+certainly+did.+I+was+I+really+enjoyed+it.+And+yeah%2C+it+was+getting+paid+to+learn+and+write+application.+So+yeah%2C+it+was+it+was+awesome.+It+was+awesome.+Yeah%2C%0Ait%27s+fantastic.+I+remember+my+first+experience+like+that+as+well.+I%27m+like%2C+I+had+better+figure+this+out+before+they+realize+I+can%27t+actually+do+this+stuff.+Yeah%2C+I+could+do+the+things+they+want.+If+I+like%2C+oh%2C+any+moment%2C+they%27re+just+gonna+say+no%2C+if+you+actually+don%27t+get+to+do+this+anymore%2C+but+it+was+it+was+great.

- Echar un vistazo a este modelo también, aunque parece peor:
https://huggingface.co/rohitsroch/hybrid_hbh_bart-base_icsi_sum?text=It%27s+not+exciting%2C+but+it%27s+the+kind+of+stuff+that+if+you+look+at+what+people+were+doing+before+was+dreadful%2C+right.+It%27s+like+transformative%2C+even+though+it+seems+so+boring+and+basic.+Yeah%2C%0Ayeah.+And+he+said%2C+Why+don%27t+you+much+come+work+for+me+and+this+consulting+firm%2C+the+main+the+owner+had+some+life+issues%2C+whatever%2C+and+had+to+move+on+and%2C+and+I+said%2C+Well%2C+I+don%27t+you+know%2C+I+don%27t+know+what+I%27m+doing.+He+said%2C+Oh%2C+we%27ll+figure+it+out.+There%27s+a+big+Barnes+and+Noble+at+the+top+of+the+hill+there+you+can+we+got+a+whole+section+on+it.+Yeah%2C+it+literally+Yeah.+And+and+I+did%2C+and+I+did+and+you+know%2C+it+was+I+had+a+passion+I+had+a+burning+desire%2C+right+to+learn.+It+was+fascinating+to+me.+Yeah.+And+so+I+switched+over+a+computer+science+degree+%27MIS%27+and+but+by+the+time+I+got+farther+along%2C+in+that+I+was+getting+paid+the+program+and+I+was+learning+a+ton+more+on+the+job.+So+that+kind+of+remember%0Ayour+first+experience+of+getting+paid+to+write+code.+Was+it+just+like%2C+I+can%27t+believe+actually+paying+me+to+do+this%3F+Absolutely.+Absolutely.+It+was.%0AYeah.+I+mean%2C+it%27s+literally+how+they+say%2C+you+know%2C+find+something+you+love+and+are+passionate+in+and+the+money+will+come.+And+yeah%2C+it+did.+It+certainly+did.+I+was+I+really+enjoyed+it.+And+yeah%2C+it+was+getting+paid+to+learn+and+write+application.+So+yeah%2C+it+was+it+was+awesome.+It+was+awesome.+Yeah%2C%0Ait%27s+fantastic.+I+remember+my+first+experience+like+that+as+well.+I%27m+like%2C+I+had+better+figure+this+out+before+they+realize+I+can%27t+actually+do+this+stuff.+Yeah%2C+I+could+do+the+things+they+want.+If+I+like%2C+oh%2C+any+moment%2C+they%27re+just+gonna+say+no%2C+if+you+actually+don%27t+get+to+do+this+anymore%2C+but+it+was+it+was+great.

- Para ver como crear un dataset:
https://huggingface.co/datasets/edinburghcstr/ami


- Descargar modelo de spacy:

```console
python -m spacy download en_core_web_sm
```
