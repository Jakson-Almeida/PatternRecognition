import numpy as np
import os
import logging
import telnetlib
import serial

logger = logging.getLogger(__name__)

class BraggMeter:
    """
    Represents a BraggMeter object used for controlling and retrieving data from a BraggMeter device.

    Args:
        host (str): The IP address of the BraggMeter device. Default is '10.0.0.150'.
        port (int): The port number of the BraggMeter device. Default is 3500.

    Attributes:
        commands (dict): A dictionary containing the commands used to communicate with the BraggMeter device.
        host (str): The IP address of the BraggMeter device.
        port (int): The port number of the BraggMeter device.
        timeout (int): The timeout value for the telnet connection.
        tn (telnetlib.Telnet): The telnet connection object.

    Raises:
        RuntimeError: If the BraggMeter device is in the heating state.

    """

    def __init__(self, host='10.0.0.150', port=3500):
        """
        Initializes a new instance of the BraggMeter class.

        Args:
            host (str): The IP address of the BraggMeter device. Default is '10.0.0.150'.
            port (int): The port number of the BraggMeter device. Default is 3500.

        Raises:
            RuntimeError: If the BraggMeter device is in the heating state.

        """
        self.commands = {'status': ":STAT?\r\n".encode('ascii'),
                         'start': ":ACQU:STAR\r\n".encode('ascii'),
                         'stop': ":ACQU:STOP\r\n".encode('ascii'),
                         'trace0': ":ACQU:OSAT:CHAN:0?\r\n".encode('ascii'),
                         'trace1': ":ACQU:OSAT:CHAN:1?\r\n".encode('ascii'),
                         'trace2': ":ACQU:OSAT:CHAN:2?\r\n".encode('ascii'),
                         'trace3': ":ACQU:OSAT:CHAN:3?\r\n".encode('ascii'),
                         'bragg0': ":ACQU:WAVE:CHAN:0?\r\n".encode('ascii'),
                         'bragg1': ":ACQU:WAVE:CHAN:1?\r\n".encode('ascii'),
                         'bragg2': ":ACQU:WAVE:CHAN:2?\r\n".encode('ascii'),
                         'bragg3': ":ACQU:WAVE:CHAN:3?\r\n".encode('ascii'),
                         'power0': ":ACQU:POWE:CHAN:0?\r\n".encode('ascii'),
                         'power1': ":ACQU:POWE:CHAN:1?\r\n".encode('ascii'),
                         'power2': ":ACQU:POWE:CHAN:2?\r\n".encode('ascii'),
                         'power3': ":ACQU:POWE:CHAN:3?\r\n".encode('ascii'),
                         'gain0?': ":ACQU:CONF:GAIN:CHAN:0?".encode('ascii'),
                         'gain1?': ":ACQU:CONF:GAIN:CHAN:1?".encode('ascii'),
                         'gain2?': ":ACQU:CONF:GAIN:CHAN:2?".encode('ascii'),
                         'gain3?': ":ACQU:CONF:GAIN:CHAN:3?".encode('ascii'),
                         }
        self.host = host
        self.port = port
        self.timeout = 10
        self.tn = telnetlib.Telnet(self.host, self.port, self.timeout)
        self.tn.close()

        status = self.get_status()
        logger.info(f"BraggMeter status: {status}")
        if status == 5:
            err_msg = 'BraggMETER em aquecimento'
            logger.error(err_msg)
            raise RuntimeError(err_msg)

    def ask(self, key):
        """
        Sends a command to the BraggMeter device and returns the response.

        Args:
            key (str): The key corresponding to the command to be sent.

        Returns:
            str: The response from the BraggMeter device.

        """
        string = self.commands[key]
        resp = self.send(string)
        return resp

    def send(self, string):
        """
        Opens a telnet connection to the BraggMeter device, sends a command, and returns the response.

        Args:
            string (str): The command to be sent.

        Returns:
            str: The response from the BraggMeter device.

        """
        self.tn.open(self.host, port=self.port)
        self.tn.write(string)
        resp = self.tn.read_until("\n".encode('ascii'), self.timeout)
        resp = resp.decode()
        logger.debug(f'{string} response: {resp}')
        self.tn.close()
        return resp

    def start(self):
        """
        Starts the acquisition process on the BraggMeter device.

        """
        status = self.get_status()
        logger.info(f'BraggMETER status: {status}')
        if status == 1:
            self.ask('start')
        elif status == 3 or status == 4:
            self.ask('stop')
            self.ask('start')
        elif status == 5:
            err_msg = 'BraggMETER em aquecimento'
            logger.error(err_msg)
            raise RuntimeError(err_msg)

    def stop(self):
        """
        Stops the acquisition process on the BraggMeter device.

        Returns:
            str: The response from the BraggMeter device.

        """
        resp = self.ask('stop')
        status = self.get_status()
        logger.info(f'BraggMETER status: {status}')
        return resp

    def get_status(self):
        """
        Retrieves the status of the BraggMeter device.

        Returns:
            int: The status code of the BraggMeter device.

        """
        resp = self.ask('status')
        logger.debug(f'Resposta do status: {resp}')
        resp = resp.split(':')
        loc = 0
        for i in range(0, len(resp)):
            if resp[i] == 'ACK':
                loc = i + 1
        return int(resp[loc])

    def get_osa_trace(self, channel):
        """
        Retrieves the OSA trace data from the specified channel.

        Args:
            channel (int): The channel number.

        Returns:
            numpy.ndarray: An array containing the wavelength and trace data.

        """
        resp = self.ask(f'trace{channel}')
        resp = resp.split(':')
        pot, wl = resp[-2], resp[-1]
        wl = np.array([float(x) for x in wl.split(',')])
        hex_values = [pot[i:i+3] for i in range(0, len(pot), 3)]
        trace = [int(hex_value, 16) for hex_value in hex_values]
        trace = np.array(trace)

        return np.append(wl.reshape(-1, 1),
                         trace.reshape(-1, 1),
                         axis=1)

    def get_peaks(self, channel):
        """
        Retrieves the peak intensity from the specified channel.

        Args:
            channel (int): The channel number.

        Returns:
            list: A list of peak intensities.

        """
        try:
            lambdas = self.ask(f'power{channel}')
        except Exception as e:
            logger.error(f'Erro ao ler intensidade do Bragg: {e}')
            self.start()
            lambdas = self.ask(f'power{channel}')
        i = lambdas.find('ACK') + 4
        lambdas = lambdas[i:-2].split(',')
        if len(lambdas) == 0:
            return []
        if lambdas[0] == '':
            return []
        return [float(lamb) if lamb else 0 for lamb in lambdas]
    
    def get_bragg(self, channel):
        """
        Retrieves the peak wavelength from the specified channel.

        Args:
            channel (int): The channel number.

        Returns:
            list: A list of peak positions.

        """
        try:
            lambdas = self.ask(f'bragg{channel}')
        except Exception as e:
            logger.error(f'Erro ao ler intensidade do Bragg: {e}')
            self.start()
            lambdas = self.ask(f'bragg{channel}')
        i = lambdas.find('ACK') + 4
        lambdas = lambdas[i:-2].split(',')
        if len(lambdas) == 0:
            return []
        if lambdas[0] == '':
            return []
        return [float(lamb) if lamb else 0 for lamb in lambdas]
    
    def set_gain(self, channel, gain):
        """
        Sets the gain for the specified channel.

        Args:
            channel (int): The channel number.
            gain (str): The gain value to be set from 0 to 255.

        Returns:
            str: The response from the BraggMeter device.

        """
        command = f":ACQU:CONF:GAIN:CHAN:{channel}:{gain}".encode('ascii')
        resp = self.send(command)
        return resp

    def get_gain(self, channel):
        """
        Retrieves the gain value for the specified channel.

        Args:
            channel (int): The channel number.

        Returns:
            str: The gain value.

        """
        resp = self.ask(f'gain{channel}?')
        return resp


class Imon512:
    """
    Represents an Imon512 object used for controlling and retrieving data from an Imon512 device.

    Args:
        port (str): The port name of the Imon512 device. Default is 'COM5'.
        baudrate (int): The baud rate of the Imon512 device. Default is 921600.

    Attributes:
        port (str): The port name of the Imon512 device.
        baudrate (int): The baud rate of the Imon512 device.
        serial_port (serial.Serial): The serial port connection object.

    """

    def __init__(self, port='COM5', baudrate=921600):
        self.port = port
        self.baudrate = baudrate

        self.serial_port = None
        self.open()

        self.ask('*idn?')
        response = self.listen()
        logger.debug(response.decode())

        # Coeficientes para o IBSEN (Patrimônio 0034/2023)
        # PN 105-1074 / SN 191382
        # A, B1, B2, ..., B5
        self.wl_param = [1.596227e3, -1.380588e-1,
                         -6.197645e-5, -5.290868e-9,
                         4.363884e-12, -3.879178e-15]
        # alpha, alpha0, beta, beta0
        self.tem_param = [1.593802e-6, -2.178398e-5,
                          -3.364313e-3, 5.350232e-2]
        self.wl = np.arange(0, 510, dtype=float)
        self.temp = None
        self.fit_wavelength(510)

    def __del__(self):
        try:
            self.close()
        except Exception as e:
            logger.error(f'Erro ao fechar porta {self.port}: {e}')

    def close(self):
        """
        Closes the serial port connection.

        """
        self.serial_port.close()
        self.serial_port = None

    def open(self):
        """
        Opens the serial port connection.

        """
        try:
            self.serial_port = serial.Serial(port=self.port,
                                             baudrate=self.baudrate)
        except Exception as e:
            logger.error(f'Erro ao abrir porta {self.port}: {e}')

    def ask(self, command):
        """
        Sends a command to the Imon512 device.

        Args:
            command (str): The command to be sent.

        """
        self.serial_port.write(command.encode())

    def listen(self):
        """
        Listens for the response from the Imon512 device.

        Returns:
            bytes: The response from the Imon512 device.

        """
        response = self.serial_port.readline()
        return response

    def fit_wavelength(self, n):
        """
        Fits the wavelength data.

        Args:
            n (int): The number of data points.

        """
        self.wl = np.arange(0, n, dtype=float)
        self.wl = self.wl_param[0] + self.wl_param[1] * self.wl + self.wl_param[2] * self.wl**2 + \
                  self.wl_param[3] * self.wl**3 + self.wl_param[4] * self.wl**4 + self.wl_param[5] * self.wl**5

    def get_temperature(self):
        """
        Retrieves the temperature data from the Imon512 device.

        Returns:
            float: The temperature value.

        """
        self.ask('temperatura?')
        response = self.listen()
        response = response.decode().strip()
        self.temp = float(response)
        return self.temp

    def get_wavelength(self):
        """
        Retrieves the wavelength data from the Imon512 device.

        Returns:
            numpy.ndarray: An array containing the wavelength data.

        """
        return self.wl

    def ask(self, command):
        """
        Send command to Imon512.

        Args:
            command (str): The command to be sent.
        """
        self.serial_port.flushInput()
        self.serial_port.flushOutput()
        serial_string = command + '\r'
        serial_string = serial_string.encode()
        self.serial_port.write(serial_string)

    def listen(self, terminator=None, n_bytes=None):
        """
        Listen incoming data from the Imon512 device.

        Args:
            terminator (bytes): The terminator for the response.
            n_bytes (int): The number of bytes to be read.

        Returns:
            str: device response.

        """
        if terminator is not None:
            response = self.serial_port.read_until(terminator)
        else:
            if n_bytes is None:
                response = self.serial_port.read(1)
                bufferBytes = self.serial_port.inWaiting()
                if bufferBytes:
                    response = response + self.serial_port.read(bufferBytes)
                return response
            else:
                if isinstance(n_bytes, int):
                    response = self.serial_port.read(n_bytes)
                else:
                    raise TypeError('n_bytes is the wrong type')
            self.serial_port.read()
        return response

    def fit_wavelength(self, n_pix):
        """
        Fit pixels to wavelength.
        Updates the wavelength attribute.

        Args:
            n_pix (int): The number of pixels.

        """
        pix = np.arange(0, n_pix, dtype=float)
        wl = np.zeros_like(pix)
        for n, coef in enumerate(self.wl_param):
            wl += coef * pix ** float(n)
        self.wl = wl
        self.temperature_compensation()

    def temperature_compensation(self):
        """
        Compensates the wavelength data for temperature.
        Updates the wavelength attribute.

        """
        self.ask('*meas:temper')
        try:
            response = self.listen()
            temp = float(response.decode().split('\t')[-1].split('\r')[0])
            self.temp = temp
            self.wl = (self.wl - self.tem_param[2] * temp - self.tem_param[2]) \
                      / (1 + self.tem_param[0] * temp + self.tem_param[1])
        except Exception as e:
            logger.error(f'erro ao compensar a temperatura: {e}')

    def measure(self, n_mean=1, return_spectrum=True):
        """
        Measure the spectrum.
        
        Args:
            n_mean (int): The number of measurements to be averaged.
            return_spectrum (bool): If True, returns the spectrum data.
        
        Returns:
            numpy.ndarray: The spectrum data.
        """
        measurements = []
        self.serial_port.flushInput()
        self.ask('*meas:fstmeas')
        for i in range(0, n_mean):
            serialString = self.serial_port.read(size=2 * 510)
            values = self.bytes2adc(serialString, n=510)
            measurements.append(values[1::])
        self.ask('esc')
        measurements = np.array(measurements)
        measurements = np.log10(measurements)

        if n_mean > 1:
            measurements = np.mean(measurements, axis=0)
        if return_spectrum:
            spectrum = np.append(self.wl[1::].reshape(-1, 1), measurements.reshape(-1, 1), axis=1)
            return np.flipud(spectrum)

    @staticmethod
    def bytes2adc(streamed_bytes, n=512):
        """
        Convert bytes to ADC values.
        
        Args:
            streamed_bytes (bytes): The streamed bytes.
            n (int): The number of ADC values.
        
        Returns:
            list: The ADC values.
        """
        values = []
        for i in range(0, 2 * n, 2):
            v = int.from_bytes(streamed_bytes[i:i + 2], byteorder='little')
            values.append(v)
        return values