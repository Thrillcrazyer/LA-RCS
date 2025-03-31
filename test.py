import YB_Pcb_Car
import time
import RPi.GPIO as GPIO

class Robot_api():
    def __init__(self):
        self.car=YB_Pcb_Car.YB_Pcb_Car()
        
        
    def getdistance(self):
        distance=0
        return distance
    
    def carback(self,distance=0):
        self.car.Car_Back(100,100)
        time.sleep(distance*2)
        self.car.Car_Stop()
    
    def carforward(self,distance=0):
        self.car.Car_Run(100,100)
        time.sleep(distance*2)
        self.car.Car_Stop()
    
    def carleft(self,angle=0):
        self.car.Car_Spin_Left(100,100)
        time.sleep(angle/180)
        self.car.Car_Stop()
    
    def carright(self,angle=0):
        self.car.Car_Spin_Right(100,100)
        time.sleep(angle/180)
        self.car.Car_Stop()
    
    def cameramove(pan=90,tilt=90):
        self.car.Servo(1,pan)
        time.sleep(0.5)
        self.car.Servo(2,tilt)
        time.sleep(0.5)
        
    def getdistance(self):
        EchoPin = 18
        TrigPin = 16
        
        GPIO.output(TrigPin,GPIO.LOW)
        time.sleep(0.000002)
        GPIO.output(TrigPin,GPIO.HIGH)
        time.sleep(0.000015)
        GPIO.output(TrigPin,GPIO.LOW)

        t3 = time.time()

        while not GPIO.input(EchoPin):
            t4 = time.time()
            if (t4 - t3) > 0.03 :
                return -1
        t1 = time.time()
        while GPIO.input(EchoPin):
            t5 = time.time()
            if(t5 - t1) > 0.03 :
                return -1

        t2 = time.time()
        time.sleep(0.01)
        #print ("distance_1 is %d " % (((t2 - t1)* 340 / 2) * 100))
        return ((t2 - t1)* 340 / 2) * 100
    
    
    