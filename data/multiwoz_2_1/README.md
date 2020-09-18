This folder contains example dialogues extracted from MultiWOZ 2.1 data processed by TRADE.

To extract a dialogue from the TRADE processed json file, you can run
```bash
jq  '.[] | select (.dialogue_idx == "MUL1626.json")' dev_dials.json
```

The [MultiWoZ 2.1 dataset](https://www.repository.cam.ac.uk/handle/1810/294507) has 
licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.
<img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" />
