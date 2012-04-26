import zlib, base64, cPickle
classifier = cPickle.loads(zlib.decompress(base64.b64decode("eJx1Vnk41O0aNigxFKks7SVNfVFSOmV5TuSjb4oJP0XLGGOYyRh+M0NJhUTS6VPJR2Vr0zJExfhSelqEUZF9zTYppQ5Kq5ajZTqnP857Xb/3vn73e9/P+/zxXtdzRyizRf58DksoMBWFBJiy+SyRiCPSWMUTDHOu7nYaJGVOJKm8g1ShEar8QJGIVCWU+QvJEQTFzolQE7FZfA7TjhwZS6iFcITegSIOqUZXIlR9gll8clQsoeXLEzN5AjFHyOYEiUn1YR1TFMQSDus09hLq3/w+TDsmSXWAfqVvi9D61gVzC4fnxxUzSU22IDggKNSUHSjkmAYE88U8llDICtVgCjnsQIFILAxmizVIre8qDYHP91NyNF0pmqB473Ehx9DolOGfHwIfcWgQR4PUJpR9/0FXolOGBTo0ugpBsXJycvrjy/D6ttGVxN57ieGNUAviCFh8cSg5ljuCoH5rgPmtQ1KXUAkMEZLjCHUhawuTHcjxZZLjuVrc/1494evVFPpADFcn1i1WTUlpRaZ+5gjLF7hS/4G+ztUKTN7fwTCaMAA6cR/tsfcyyP/591ZGfQcI+4d6vTVuYJy5r7vN3gJYa9jaFC65A9UNSafM56XjS6O4molF1zG7PbLF9s1d2H1w979GDJTAnOdlVxbYFgEnzuSihCkHP3lovfesUiiR/rlT6vwWQ6VLxvbNeYSnNtDjJyQ/xkirAwbRSx9ghPy4k9GXOnw1LvSOgfQxPKdsDa9r3gDbmOyXJvadOJNO1a+bkoO1QntRwdBFKD8bM6NX6zqayW0LY5IkaH3eeuV62VPgql4pMtdvQG3to9dEle1YcC5K2OYxgDtXSL0KIj+hJIxPr5l2Eh5jBoWhXYp2UZt3HHdNhHCfvoWrk8tRaecm3QtrcjF+g9T5QmwZGuoJTufx5RhJNTJbb5WPMyisfQmHq5Dr/HviEuc2rC30GOmsXIU56jO2mRIx0FF/tvPP/mr8y3+QOmVuLd4Nnhj4+slDXLzOYrCbK8db/drlWXuacS0xrzF2WRma2xzP52V1gl/kidaTWfVQkqXlUjuYBfkWln2Ll+XB7kT5Rdr6MhBu1O+PbL4NOx4tarM5XIjL9lvdVva4A8K+fuqaovuQSo59ZuZ+H7JeJa9I338TPpPzD+iWFMHY1iXGuvJHuDecefsgtQs5Baf0suZ04LaWe4QhdkF93IJu92tyeLd1VhLT/SwUxTMkuw2qoZL1fnKYvBtkDu56o2kF6MU3a7j+ugQNczQnTZ7WiQ5eTL2pLtnYvWVmT9aJk/g8Q6tl94s61D4bbDPAyYBVPdbLX2oWg2iX4H6jZhl0299k9bxux1lZa1JWq5aB/VHXzoR1MpjYIJj0IikCr8bJ/rJ41YqMppZBunMLLkXfa4HuySiXOfnXHqpEiaHQxCMqEQj6FY+wjTkoZBTLFiwvgFYVgl6sfw8ILVawVUIePluyz8t88Sk8NL49xH3cEVihuezAi+hsqNYo0L+jIsEkhtf5RHyD/+ZVhNs+bUJDP1qP/8smOL73wPTDOo/Qk5o8U6+6DfqC7YayX1ZCP01nZ+vpqyB5W0pTe14GZgYekrapecAs8q+NiszGaTlHP16jnoHZUT4f0hybIcWYXjh35iV832KblCJrh4NGo8VTSxtg95uGi+8XvMRGTVehbc11KEq+UvabVSdoSiWtyYIOON24+nCOuAli/CvKB0JugX/eh4a0fcXQ5Tv1s/Qo4k3Z9JUT39XgzT0mi2Qp5Wh5b/vMIGkn8GwuLByq7MBProUsSJbCteNuKvHWN2DbJt6yMzpNENF9xEId20FPm5wcuvYR0D6r9EVb5ILn/Nvm090KoEX22XXX20IIefbFkWNRCE4rFxr3nD2D5wpoFLXUHFyTRw0z2C5FyM8JqBXWYvQI2rK02lvI77bM5a0uh9TD4SWOXyrwXJ2vvZPkOjjmrNOlPisB7djFEVGZ94DLvpk+Q68RzOoz3gSnNYDjJON0g0116EFWPTtlUIm5bqqbTRglcOcSkdRDlv5EBa/QKXzv21IZyiH5sMQmUOWPmO2Yd9PTYOhWLYZPtltw1LIIFKjgFTqFb/DYLPcp6pVw8DnhyBp/H0al5VxNc70BnOt3qZHRxT9RwSt0Ct/sCdNX+917gg22NZdcJ6bDLuXRXKuNzcCJyhscLbv1ExW8QqfwndicqfzbhS4g1zwt0KHfhkW8+exPLQlwd8eID89u1fxEBa/QKXwbts6SHdJOBbfsEzrtJVX4araLxpzcQmwZWvJWafs1UKCCV+gUvkcvQrN3hl0FgVJdsCftAcQvb3SIe3MZ1jb8nstIqQL1Hp0zneMbcUaTls3m2C5Ui59SFHPuIdqwcyUhrHPYOvfhBc9P1Tgh7PXCvr0V+EpQeCwqoAGjQk2Pk0U1sPnYzOVPOrqhjSoJMJn9AhIuy13e2fTg4mmriFy/qxiV5X65btwA1PM/PqfqlWHuvMGMqR+zht+VMYoSSuDw0nsjqyKbUNtK62S0uAY8pUayXuuTYFatL5i6pgzX2SWIvcaVwjGjVjePPcWoa6C0xyXsDrT3L9CN39SFQZm9FGavF8am1a36YNwMufnjuiUOxdjhOSVm+8B5HIXG+eaZ7fAh8zPXknkR36jbj8GceFQ2mGGZof43jHlfbFXaX4Y2CyNUc2nlMGmdjo/1sXSoeJpCfRjuD/QnlUGM+SWgdupIq2N4Cl4cGl0VcKkQEi9lJr660oS8jI0Vs4wScZ90386T+amY6KS+1npLNvYy9ugzmvPBAUOnpZ55h8NzXUUcyCf1HMDQoqt3kp0JMZLP8ubwmaT+LxPc4OsEV47mDucF3qIfecHw/+aFUV8DDGX4G66v+0uM+VF84i/FJ31PJlzD70lD838d5GQnQudncmJ+TUo8gR85hU4J9jb9D+PFnVM=")))