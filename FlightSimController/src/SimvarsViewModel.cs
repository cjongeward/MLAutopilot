using System;
using System.Windows;
using System.Collections.ObjectModel;
using System.Globalization;
using System.IO;
using System.Runtime.InteropServices;
using System.Windows.Threading;

using Microsoft.FlightSimulator.SimConnect;
using System.Collections.Generic;
using System.Collections;

using Python.Runtime;

namespace Simvars
{
    public enum DEFINITION
    {
        Dummy = 0
    };

    public enum REQUEST
    {
        Dummy = 0,
        Struct1
    };

    public enum COPY_ITEM
    {
        Name = 0,
        Value,
        Unit
    }

    // String properties must be packed inside of a struct
    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi, Pack = 1)]
    struct Struct1
    {
        // this is how you declare a fixed size string
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 256)]
        public String sValue;

        // other definitions can be added to this struct
        // ...
    };

    public class SimvarRequest : ObservableObject
    {
        public DEFINITION eDef = DEFINITION.Dummy;
        public REQUEST eRequest = REQUEST.Dummy;

        public string sName { get; set; }
        public bool bIsString { get; set; }
        public double dValue
        {
            get { return m_dValue; }
            set { this.SetProperty(ref m_dValue, value); }
        }
        private double m_dValue = 0.0;
        public string sValue
        {
            get { return m_sValue; }
            set { this.SetProperty(ref m_sValue, value); }
        }
        private string m_sValue = null;

        public string sUnits { get; set; }

        public bool bPending = true;
        public bool bStillPending
        {
            get { return m_bStillPending; }
            set { this.SetProperty(ref m_bStillPending, value); }
        }
        private bool m_bStillPending = false;

    };

    public class SimvarsViewModel : BaseViewModel, IBaseSimConnectWrapper
    {
        #region IBaseSimConnectWrapper implementation

        /// User-defined win32 event
        public const int WM_USER_SIMCONNECT = 0x0402;

        /// Window handle
        private IntPtr m_hWnd = new IntPtr(0);

        /// SimConnect object
        private SimConnect m_oSimConnect = null;

        public bool bConnected
        {
            get { return m_bConnected; }
            private set { this.SetProperty(ref m_bConnected, value); }
        }
        private bool m_bConnected = false;

        private uint m_iCurrentDefinition = 0;
        private uint m_iCurrentRequest = 0;

        public int GetUserSimConnectWinEvent()
        {
            return WM_USER_SIMCONNECT;
        }

        public void ReceiveSimConnectMessage()
        {
            m_oSimConnect?.ReceiveMessage();
        }

        public void SetWindowHandle(IntPtr _hWnd)
        {
            m_hWnd = _hWnd;
        }

        public void Disconnect()
        {
            Console.WriteLine("Disconnect");
            PythonEngine.Shutdown();

            m_oTimer.Stop();
            bOddTick = false;

            if (m_oSimConnect != null)
            {
                /// Dispose serves the same purpose as SimConnect_Close()
                m_oSimConnect.Dispose();
                m_oSimConnect = null;
            }

            sConnectButtonLabel = "Connect";
            bConnected = false;

            // Set all requests as pending
            foreach (SimvarRequest oSimvarRequest in lSimvarRequests)
            {
                oSimvarRequest.bPending = true;
                oSimvarRequest.bStillPending = true;
            }
        }
        #endregion

        #region UI bindings

        public string sConnectButtonLabel
        {
            get { return m_sConnectButtonLabel; }
            private set { this.SetProperty(ref m_sConnectButtonLabel, value); }
        }
        private string m_sConnectButtonLabel = "Connect";
        public string sCollectingModeButtonLabel
        {
            get { return m_sCollectingModeButtonLabel; }
            private set { this.SetProperty(ref m_sCollectingModeButtonLabel, value); }
        }
        private string m_sCollectingModeButtonLabel = "Collecting off";
        public string sFlyingModeButtonLabel
        {
            get { return m_sFlyingModeButtonLabel; }
            private set { this.SetProperty(ref m_sFlyingModeButtonLabel, value); }
        }
        private string m_sFlyingModeButtonLabel = "Flying off";




        public bool bObjectIDSelectionEnabled
        {
            get { return m_bObjectIDSelectionEnabled; }
            set { this.SetProperty(ref m_bObjectIDSelectionEnabled, value); }
        }
        private bool m_bObjectIDSelectionEnabled = false;
        public SIMCONNECT_SIMOBJECT_TYPE eSimObjectType
        {
            get { return m_eSimObjectType; }
            set
            {
                this.SetProperty(ref m_eSimObjectType, value);
                bObjectIDSelectionEnabled = (m_eSimObjectType != SIMCONNECT_SIMOBJECT_TYPE.USER);
                ClearResquestsPendingState();
            }
        }
        private SIMCONNECT_SIMOBJECT_TYPE m_eSimObjectType = SIMCONNECT_SIMOBJECT_TYPE.USER;
        public ObservableCollection<uint> lObjectIDs { get; private set; }
        public uint iObjectIdRequest
        {
            get { return m_iObjectIdRequest; }
            set
            {
                this.SetProperty(ref m_iObjectIdRequest, value);
                ClearResquestsPendingState();
            }
        }
        private uint m_iObjectIdRequest = 0;

        public string[] aSimvarNames
        {
            get { return SimUtils.SimVars.Names; }
            set { this.SetProperty(ref m_aSimvarNames, value); }
        }
        private string[] m_aSimvarNames = null;

        public string[] aSimvarNamesFiltered
        {
            get
            {
                if (m_aSimvarNamesFiltered == null)
                {
                    m_aSimvarNamesFiltered = aSimvarNames;
                }
                return m_aSimvarNamesFiltered;
            }
            set { this.SetProperty(ref m_aSimvarNamesFiltered, value); }
        }
        private string[] m_aSimvarNamesFiltered = null;

        public string sSimvarRequest
        {
            get { return m_sSimvarRequest; }
            set { this.SetProperty(ref m_sSimvarRequest, value); }
        }
        private string m_sSimvarRequest = null;

        public string[] aUnitNames
        {
            get
            {
                if (m_aUnitNames == null)
                {
                    m_aUnitNames = SimUtils.Units.Names;
                    Array.Sort(m_aUnitNames);
                }
                return m_aUnitNames;
            }
            private set { }
        }
        private string[] m_aUnitNames;
        public string[] aUnitNamesFiltered
        {
            get
            {
                if (m_aUnitNamesFiltered == null)
                {
                    m_aUnitNamesFiltered = aUnitNames;
                }
                return m_aUnitNamesFiltered;
            }
            set { this.SetProperty(ref m_aUnitNamesFiltered, value); }
        }
        private string[] m_aUnitNamesFiltered = null;


        public string sUnitRequest
        {
            get { return m_sUnitRequest; }
            set { this.SetProperty(ref m_sUnitRequest, value); }
        }
        private string m_sUnitRequest = null;

        public string sSetValue
        {
            get { return m_sSetValue; }
            set { this.SetProperty(ref m_sSetValue, value); }
        }
        private string m_sSetValue = null;
        public string sDHdg
        {
            get { return m_sDHdg; }
            set { this.SetProperty(ref m_sDHdg, value); }
        }
        private string m_sDHdg = null;
        public string sDAlt
        {
            get { return m_sDAlt; }
            set { this.SetProperty(ref m_sDAlt, value); }
        }
        private string m_sDAlt = null;
        public string sDIas
        {
            get { return m_sDIas; }
            set { this.SetProperty(ref m_sDIas, value); }
        }
        private string m_sDIas = null;

        public ObservableCollection<SimvarRequest> lSimvarRequests { get; private set; }
        public SimvarRequest oSelectedSimvarRequest
        {
            get { return m_oSelectedSimvarRequest; }
            set { this.SetProperty(ref m_oSelectedSimvarRequest, value); }
        }
        private SimvarRequest m_oSelectedSimvarRequest = null;

        public uint[] aIndices
        {
            get { return m_aIndices; }
            private set { }
        }
        private readonly uint[] m_aIndices = new uint[100] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                                                                                                                        10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                                                                                                                        20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                                                                                                                        30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                                                                                                                        40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
                                                                                                                        50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
                                                                                                                        60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
                                                                                                                        70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
                                                                                                                        80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
                                                                                                                        90, 91, 92, 93, 94, 95, 96, 97, 98, 99 };
        public uint iIndexRequest
        {
            get { return m_iIndexRequest; }
            set { this.SetProperty(ref m_iIndexRequest, value); }
        }
        private uint m_iIndexRequest = 0;

        public bool bSaveValues
        {
            get { return m_bSaveValues; }
            set { this.SetProperty(ref m_bSaveValues, value); }
        }
        private bool m_bSaveValues = true;

        public bool bFSXcompatible
        {
            get { return m_bFSXcompatible; }
            set { this.SetProperty(ref m_bFSXcompatible, value); }
        }
        private bool m_bFSXcompatible = false;
        public bool bIsString
        {
            get { return m_bIsString; }
            set { this.SetProperty(ref m_bIsString, value); }
        }
        private bool m_bIsString = false;

        public bool bShowAllUnits
        {
            get { return m_bShowAllUnits; }
            set { this.SetProperty(ref m_bShowAllUnits, value); }
        }
        private bool m_bShowAllUnits = false;

        public bool bOddTick
        {
            get { return m_bOddTick; }
            set { this.SetProperty(ref m_bOddTick, value); }
        }
        private bool m_bOddTick = false;

        public ObservableCollection<string> lErrorMessages { get; private set; }


        public BaseCommand cmdToggleConnect { get; private set; }
        public BaseCommand cmdToggleCollectingMode { get; private set; }
        public BaseCommand cmdToggleFlyingMode { get; private set; }
        public BaseCommand cmdSaveData { get; private set; }
        public BaseCommand cmdClearData { get; private set; }
        public BaseCommand cmdAddRequest { get; private set; }
        public BaseCommand cmdRemoveSelectedRequest { get; private set; }
        public BaseCommand cmdRemoveAllRequests { get; private set; }
        public BaseCommand cmdCopyNameSelectedRequest { get; private set; }
        public BaseCommand cmdCopyValueSelectedRequest { get; private set; }
        public BaseCommand cmdCopyUnitSelectedRequest { get; private set; }
        public BaseCommand cmdTrySetValue { get; private set; }
        public BaseCommand cmdLoadFiles { get; private set; }
        public BaseCommand cmdSaveFile { get; private set; }
        public BaseCommand cmdSaveFileWithValues { get; private set; }

        #endregion

        #region Real time

        private DispatcherTimer m_oTimer = new DispatcherTimer();

        #endregion

        public SimvarsViewModel()
        {
            lObjectIDs = new ObservableCollection<uint>();
            lObjectIDs.Add(1);

            lSimvarRequests = new ObservableCollection<SimvarRequest>();

            lErrorMessages = new ObservableCollection<string>();

            cmdToggleConnect = new BaseCommand((p) => { ToggleConnect(); });
            cmdToggleCollectingMode = new BaseCommand((p) => { toggleCollectingMode(); });
            cmdToggleFlyingMode = new BaseCommand((p) => { toggleFlyingMode(); });
            cmdSaveData = new BaseCommand((p) => { SaveData(); });
            cmdClearData = new BaseCommand((p) => { ClearData(); });
            cmdAddRequest = new BaseCommand((p) => { AddRequest((m_iIndexRequest == 0) ? m_sSimvarRequest : (m_sSimvarRequest + ":" + m_iIndexRequest), sUnitRequest, bIsString); });
            cmdRemoveSelectedRequest = new BaseCommand((p) => { RemoveSelectedRequest(); });
            cmdRemoveAllRequests = new BaseCommand((p) => { RemoveAllRequest(); });
            cmdCopyNameSelectedRequest = new BaseCommand((p) => { CopySelectedRequest(COPY_ITEM.Name); });
            cmdCopyValueSelectedRequest = new BaseCommand((p) => { CopySelectedRequest(COPY_ITEM.Value); });
            cmdCopyUnitSelectedRequest = new BaseCommand((p) => { CopySelectedRequest(COPY_ITEM.Unit); });
            cmdTrySetValue = new BaseCommand((p) => { TrySetValue(); });
            cmdLoadFiles = new BaseCommand((p) => { LoadFiles(); });
            cmdSaveFile = new BaseCommand((p) => { SaveFile(false, false); });
            cmdSaveFileWithValues = new BaseCommand((p) => { SaveFile(false, true); });

            m_oTimer.Interval = new TimeSpan(0, 0, 0, 0, 50);
            m_oTimer.Tick += new EventHandler(OnTick);
        }

        private void Connect()
        {
            Console.WriteLine("Connect");

            try
            {
                /// The constructor is similar to SimConnect_Open in the native API
                m_oSimConnect = new SimConnect("Simconnect - Simvar test", m_hWnd, WM_USER_SIMCONNECT, null, bFSXcompatible ? (uint)1 : 0);

                /// Listen to connect and quit msgs
                m_oSimConnect.OnRecvOpen += new SimConnect.RecvOpenEventHandler(SimConnect_OnRecvOpen);
                m_oSimConnect.OnRecvQuit += new SimConnect.RecvQuitEventHandler(SimConnect_OnRecvQuit);

                /// Listen to exceptions
                m_oSimConnect.OnRecvException += new SimConnect.RecvExceptionEventHandler(SimConnect_OnRecvException);

                /// Catch a simobject data request
                m_oSimConnect.OnRecvSimobjectDataBytype += new SimConnect.RecvSimobjectDataBytypeEventHandler(SimConnect_OnRecvSimobjectDataBytype);
            }
            catch (COMException ex)
            {
                Console.WriteLine("Connection to KH failed: " + ex.Message);
            }
        }

        private void SimConnect_OnRecvOpen(SimConnect sender, SIMCONNECT_RECV_OPEN data)
        {
            Console.WriteLine("SimConnect_OnRecvOpen");
            Console.WriteLine("Connected to KH");

            sConnectButtonLabel = "Disconnect";
            bConnected = true;

            // Register pending requests
            foreach (SimvarRequest oSimvarRequest in lSimvarRequests)
            {
                if (oSimvarRequest.bPending)
                {
                    oSimvarRequest.bPending = !RegisterToSimConnect(oSimvarRequest);
                    oSimvarRequest.bStillPending = oSimvarRequest.bPending;
                }
            }

            m_oTimer.Start();
            bOddTick = false;
        }

        /// The case where the user closes game
        private void SimConnect_OnRecvQuit(SimConnect sender, SIMCONNECT_RECV data)
        {
            Console.WriteLine("SimConnect_OnRecvQuit");
            Console.WriteLine("KH has exited");

            Disconnect();
        }

        private void SimConnect_OnRecvException(SimConnect sender, SIMCONNECT_RECV_EXCEPTION data)
        {
            SIMCONNECT_EXCEPTION eException = (SIMCONNECT_EXCEPTION)data.dwException;
            Console.WriteLine("SimConnect_OnRecvException: " + eException.ToString());

            lErrorMessages.Add("SimConnect : " + eException.ToString());
        }

        private void SimConnect_OnRecvSimobjectDataBytype(SimConnect sender, SIMCONNECT_RECV_SIMOBJECT_DATA_BYTYPE data)
        {
            //Console.WriteLine("SimConnect_OnRecvSimobjectDataBytype");

            uint iRequest = data.dwRequestID;
            uint iObject = data.dwObjectID;
            if (!lObjectIDs.Contains(iObject))
            {
                lObjectIDs.Add(iObject);
            }
            foreach (SimvarRequest oSimvarRequest in lSimvarRequests)
            {
                if (iRequest == (uint)oSimvarRequest.eRequest && (!bObjectIDSelectionEnabled || iObject == m_iObjectIdRequest))
                {
                    if (oSimvarRequest.bIsString)
                    {
                        Struct1 result = (Struct1)data.dwData[0];
                        oSimvarRequest.dValue = 0;
                        oSimvarRequest.sValue = result.sValue;
                    }
                    else
                    {
                        double dValue = (double)data.dwData[0];
                        oSimvarRequest.dValue = dValue;
                        oSimvarRequest.sValue = dValue.ToString("F9");

                    }

                    oSimvarRequest.bPending = false;
                    oSimvarRequest.bStillPending = false;
                }
            }
        }

        // My variables
        enum SimVarsEnum
        {
            bank,
            heading,
            pitch,
            altitude,
            vs,
            airspeed,
            aileron_pos,
            elevator_pos,
            rudder_pos,
            throttle_pos
        };
        Dictionary<SimVarsEnum, SimvarRequest> SimVars;

        enum ModeEnum
        {
            off,
            collecting,
            flying
        };
        ModeEnum Mode;

        private void setModeStatusText()
        {
            sCollectingModeButtonLabel = "Collecting off";
            sFlyingModeButtonLabel = "Flying off";
            switch(Mode)
            {
                case ModeEnum.collecting:
                    sCollectingModeButtonLabel = "Collecting on";
                    break;
                case ModeEnum.flying:
                    sFlyingModeButtonLabel = "Flying on";
                    break;
            }

        }
        private void toggleCollectingMode()
        {
            if(Mode == ModeEnum.off)
            {
                Mode = ModeEnum.collecting;
            }
            else
            {
                Mode = ModeEnum.off;
            }
            setModeStatusText();
        }
        private void toggleFlyingMode()
        {
            if(Mode == ModeEnum.off)
            {
                Mode = ModeEnum.flying;
            }
            else
            {
                Mode = ModeEnum.off;
            }
            setModeStatusText();
        }

        struct DataEntry
        {
            public double ias;
            public double alt;
            public double hdg;
            public double dias;
            public double dalt;
            public double dhdg;
            public double bank;
            public double pitch;
            public double aileronPos;
            public double elevatorPos;
            public double rudderPos;
            public double throttlePos;

            public double dbank;
            public double dpitch;
            public double vs;
            public double dvs;
        };
        List<DataEntry> Data;

        private void init()
        {
            Environment.SetEnvironmentVariable("PYTHONNET_PYDLL", "python312.dll");
            PythonEngine.Initialize();
            using (Py.GIL()) // Acquires the Global Interpreter Lock
            {
                dynamic sys = Py.Import("sys");
                sys.path.append(@"C:\repos\MLAutopilot\scripts");
                dynamic pyModule = Py.Import("predictor"); 
            }
            SimVars = new Dictionary<SimVarsEnum, SimvarRequest>();
            Data = new List<DataEntry>();
            foreach (SimvarRequest oSimvarRequest in lSimvarRequests)
            {
                switch(oSimvarRequest.sName)
                {
                    case "HEADING INDICATOR":
                        SimVars.Add(SimVarsEnum.heading, oSimvarRequest);
                        break;
                    case "INDICATED ALTITUDE":
                        SimVars.Add(SimVarsEnum.altitude, oSimvarRequest);
                        break;
                    case "AIRSPEED INDICATED":
                        SimVars.Add(SimVarsEnum.airspeed, oSimvarRequest);
                        break;
                    case "VERTICAL SPEED":
                        SimVars.Add(SimVarsEnum.vs, oSimvarRequest);
                        break;
                    case "AILERON POSITION":
                        SimVars.Add(SimVarsEnum.aileron_pos, oSimvarRequest);
                        break;
                    case "ELEVATOR POSITION":
                        SimVars.Add(SimVarsEnum.elevator_pos, oSimvarRequest);
                        break;
                    case "RUDDER POSITION":
                        SimVars.Add(SimVarsEnum.rudder_pos, oSimvarRequest);
                        break;
                    case "GENERAL ENG THROTTLE LEVER POSITION:1":
                        SimVars.Add(SimVarsEnum.throttle_pos, oSimvarRequest);
                        break;
                    case "ATTITUDE INDICATOR BANK DEGREES":
                        SimVars.Add(SimVarsEnum.bank, oSimvarRequest);
                        break;
                    case "ATTITUDE INDICATOR PITCH DEGREES":
                        SimVars.Add(SimVarsEnum.pitch, oSimvarRequest);
                        break;
                }
            }
            Mode = ModeEnum.off;
        }

        private bool isValid()
        {
            return SimVars != null &&
                SimVars.ContainsKey(SimVarsEnum.aileron_pos) &&
                SimVars.ContainsKey(SimVarsEnum.airspeed) &&
                SimVars.ContainsKey(SimVarsEnum.altitude) &&
                SimVars.ContainsKey(SimVarsEnum.bank) &&
                SimVars.ContainsKey(SimVarsEnum.elevator_pos) &&
                SimVars.ContainsKey(SimVarsEnum.heading) &&
                SimVars.ContainsKey(SimVarsEnum.pitch) &&
                SimVars.ContainsKey(SimVarsEnum.rudder_pos) &&
                SimVars.ContainsKey(SimVarsEnum.throttle_pos) &&
                SimVars.ContainsKey(SimVarsEnum.vs);
        }

        private void OnTick(object sender, EventArgs e)
        {
            //Console.WriteLine("OnTick");

            bOddTick = !bOddTick;

            foreach (SimvarRequest oSimvarRequest in lSimvarRequests)
            {
                if (!oSimvarRequest.bPending)
                {
                    m_oSimConnect?.RequestDataOnSimObjectType(oSimvarRequest.eRequest, oSimvarRequest.eDef, 0, m_eSimObjectType);
                    oSimvarRequest.bPending = true;
                }
                else
                {
                    oSimvarRequest.bStillPending = true;
                }
            }
            if(isValid())
            {
                switch(Mode)
                {
                    case ModeEnum.collecting:
                        DataEntry entry = new DataEntry();
                        entry.aileronPos = SimVars[SimVarsEnum.aileron_pos].dValue;
                        entry.alt = SimVars[SimVarsEnum.altitude].dValue;
                        entry.bank = SimVars[SimVarsEnum.bank].dValue;
                        entry.elevatorPos = SimVars[SimVarsEnum.elevator_pos].dValue;
                        entry.hdg = SimVars[SimVarsEnum.heading].dValue;
                        entry.ias = SimVars[SimVarsEnum.airspeed].dValue;
                        entry.pitch = SimVars[SimVarsEnum.pitch].dValue;
                        entry.rudderPos = SimVars[SimVarsEnum.rudder_pos].dValue;
                        entry.throttlePos = SimVars[SimVarsEnum.throttle_pos].dValue;
                        entry.vs = SimVars[SimVarsEnum.vs].dValue;

                        double dValue = 0.0;
                        if (m_sDHdg != null && double.TryParse(m_sDHdg, NumberStyles.Any, null, out dValue))
                        {
                            entry.dhdg = dValue;
                        }
                        if (m_sDAlt != null && double.TryParse(m_sDAlt, NumberStyles.Any, null, out dValue))
                        {
                            entry.dalt = 0.0;
                        }
                        if (m_sDIas != null && double.TryParse(m_sDIas, NumberStyles.Any, null, out dValue))
                        {
                            entry.dias = 0.0;
                        }
                        entry.dbank = 0.0;
                        entry.dpitch = 0.0;
                        entry.dvs = 0.0;
                        Data.Add(entry);
                        break;

                    case ModeEnum.flying:
                        double bank = SimVars[SimVarsEnum.bank].dValue;
                        double hdg = SimVars[SimVarsEnum.heading].dValue;
                        double dhdg = 0.0;
                        double.TryParse(m_sDHdg, NumberStyles.Any, null, out dhdg);
                        using (Py.GIL()) // Acquires the Global Interpreter Lock
                        {
                            dynamic sys = Py.Import("sys");
                            sys.path.append(@"C:\repos\MLAutopilot\scripts");
                            dynamic pyModule = Py.Import("predictor"); 
                            dynamic newAileron = pyModule.predict(hdg, dhdg, bank); // Call the Python function
                            m_oSimConnect.SetDataOnSimObject(SimVars[SimVarsEnum.aileron_pos].eDef, m_iObjectIdRequest, SIMCONNECT_DATA_SET_FLAG.DEFAULT, (double)newAileron);
                        }

                        //m_oSimConnect.SetDataOnSimObject(SimVars[SimVarsEnum.aileron_pos].eDef, m_iObjectIdRequest, SIMCONNECT_DATA_SET_FLAG.DEFAULT, newAileron);
                        break;
                }
            }
        }

        private void SaveData()
        {
            Microsoft.Win32.SaveFileDialog oSaveFileDialog = new Microsoft.Win32.SaveFileDialog();
            oSaveFileDialog.Filter = "csv files (*.csv)|*.csv";
            if (oSaveFileDialog.ShowDialog() == true)
            {
                using (StreamWriter oStreamWriter = new StreamWriter(oSaveFileDialog.FileName, false))
                {
                    foreach (DataEntry entry in Data)
                    {
                        string sFormatedLine = entry.ias + "," + entry.alt + "," + entry.hdg + "," +  entry.dias + "," + entry.dalt + "," + entry.dhdg + "," + entry.bank + "," + entry.pitch + "," + entry.aileronPos + "," + entry.elevatorPos + "," + entry.rudderPos + "," + entry.throttlePos;
                        oStreamWriter.WriteLine(sFormatedLine);
                    }
                }
            }
        }

        private void ClearData()
        {
            if(Data != null)
            {
                Data.Clear();
            }
        }



        private void ToggleConnect()
        {
            if (m_oSimConnect == null)
            {
                try
                {
                    Connect();
                }
                catch (COMException ex)
                {
                    Console.WriteLine("Unable to connect to KH: " + ex.Message);
                }
            }
            else
            {
                Disconnect();
            }
        }

        private void ClearResquestsPendingState()
        {
            foreach (SimvarRequest oSimvarRequest in lSimvarRequests)
            {
                oSimvarRequest.bPending = false;
                oSimvarRequest.bStillPending = false;
            }
        }

        private bool RegisterToSimConnect(SimvarRequest _oSimvarRequest)
        {
            if (m_oSimConnect != null)
            {
                if (_oSimvarRequest.bIsString)
                {
                    /// Define a data structure containing string value
                    m_oSimConnect.AddToDataDefinition(_oSimvarRequest.eDef, _oSimvarRequest.sName, "", SIMCONNECT_DATATYPE.STRING256, 0.0f, SimConnect.SIMCONNECT_UNUSED);
                    /// IMPORTANT: Register it with the simconnect managed wrapper marshaller
                    /// If you skip this step, you will only receive a uint in the .dwData field.
                    m_oSimConnect.RegisterDataDefineStruct<Struct1>(_oSimvarRequest.eDef);
                }
                else
                {
                    /// Define a data structure containing numerical value
                    m_oSimConnect.AddToDataDefinition(_oSimvarRequest.eDef, _oSimvarRequest.sName, _oSimvarRequest.sUnits, SIMCONNECT_DATATYPE.FLOAT64, 0.0f, SimConnect.SIMCONNECT_UNUSED);
                    /// IMPORTANT: Register it with the simconnect managed wrapper marshaller
                    /// If you skip this step, you will only receive a uint in the .dwData field.
                    m_oSimConnect.RegisterDataDefineStruct<double>(_oSimvarRequest.eDef);
                }

                return true;
            }
            else
            {
                return false;
            }
        }

        private void AddRequest(string _sNewSimvarRequest, string _sNewUnitRequest, bool _bIsString)
        {
            Console.WriteLine("AddRequest");

            //string sNewSimvarRequest = _sOverrideSimvarRequest != null ? _sOverrideSimvarRequest : ((m_iIndexRequest == 0) ? m_sSimvarRequest : (m_sSimvarRequest + ":" + m_iIndexRequest));
            //string sNewUnitRequest = _sOverrideUnitRequest != null ? _sOverrideUnitRequest : m_sUnitRequest;
            SimvarRequest oSimvarRequest = new SimvarRequest
            {
                eDef = (DEFINITION)m_iCurrentDefinition,
                eRequest = (REQUEST)m_iCurrentRequest,
                sName = _sNewSimvarRequest,
                bIsString = _bIsString,
                sUnits = _bIsString ? null : _sNewUnitRequest
            };

            oSimvarRequest.bPending = !RegisterToSimConnect(oSimvarRequest);
            oSimvarRequest.bStillPending = oSimvarRequest.bPending;

            lSimvarRequests.Add(oSimvarRequest);

            ++m_iCurrentDefinition;
            ++m_iCurrentRequest;
        }

        private void RemoveSelectedRequest()
        {
            lSimvarRequests.Remove(oSelectedSimvarRequest);
        }
        private void RemoveAllRequest()
        {
            while (lSimvarRequests.Count > 0)
            {
                lSimvarRequests.RemoveAt(lSimvarRequests.Count - 1);
            }
        }

        private void CopySelectedRequest(COPY_ITEM item)
        {
            if (oSelectedSimvarRequest != null)
            {
                switch (item)
                {
                    case COPY_ITEM.Name:
                        Clipboard.SetText(oSelectedSimvarRequest.sName);
                        break;
                    case COPY_ITEM.Value:
                        Clipboard.SetText(oSelectedSimvarRequest.sValue);
                        break;
                    case COPY_ITEM.Unit:
                        Clipboard.SetText(oSelectedSimvarRequest.sUnits);
                        break;
                }
            }
        }

        private void TrySetValue()
        {
            //            Console.WriteLine("TrySetValue");
            if (m_oSelectedSimvarRequest != null && m_sSetValue != null)
            {
                if (!m_oSelectedSimvarRequest.bIsString)
                {
                    double dValue = 0.0;
                    if (double.TryParse(m_sSetValue, NumberStyles.Any, null, out dValue))
                    {
                        m_oSimConnect.SetDataOnSimObject(m_oSelectedSimvarRequest.eDef, m_iObjectIdRequest, SIMCONNECT_DATA_SET_FLAG.DEFAULT, dValue);
                    }
                }
                else
                {
                    Struct1 sValueStruct = new Struct1()
                    {
                        sValue = m_sSetValue
                    };
                    m_oSimConnect.SetDataOnSimObject(m_oSelectedSimvarRequest.eDef, m_iObjectIdRequest, SIMCONNECT_DATA_SET_FLAG.DEFAULT, sValueStruct);
                }
            }
        }

        private void LoadFiles()
        {
            Microsoft.Win32.OpenFileDialog oOpenFileDialog = new Microsoft.Win32.OpenFileDialog();
            oOpenFileDialog.Multiselect = true;
            oOpenFileDialog.Filter = "Simvars files (*.simvars)|*.simvars";
            if (oOpenFileDialog.ShowDialog() == true)
            {
                foreach (string sFilename in oOpenFileDialog.FileNames)
                {
                    LoadFile(sFilename);
                }
            }
        }

        private void LoadFile(string _sFileName)
        {
            string[] aLines = System.IO.File.ReadAllLines(_sFileName);
            for (uint i = 0; i < aLines.Length; ++i)
            {
                // Format : Simvar,Unit
                string[] aSubStrings = aLines[i].Split(',');
                if (aSubStrings.Length >= 2) // format check
                {
                    // values check
                    string[] aSimvarSubStrings = aSubStrings[0].Split(':'); // extract Simvar name from format Simvar:Index
                    string sSimvarName = Array.Find(SimUtils.SimVars.Names, s => s == aSimvarSubStrings[0]);
                    string sUnitName = Array.Find(SimUtils.Units.Names, s => s == aSubStrings[1]);
                    bool bIsString = aSubStrings.Length > 2 && bool.Parse(aSubStrings[2]);
                    if (sSimvarName != null && (sUnitName != null || bIsString))
                    {
                        AddRequest(aSubStrings[0], sUnitName, bIsString);
                    }
                    else
                    {
                        if (sSimvarName == null)
                        {
                            lErrorMessages.Add("l." + i.ToString() + " Wrong Simvar name : " + aSubStrings[0]);
                        }
                        if (sUnitName == null)
                        {
                            lErrorMessages.Add("l." + i.ToString() + " Wrong Unit name : " + aSubStrings[1]);
                        }
                    }
                }
                else
                {
                    lErrorMessages.Add("l." + i.ToString() + " Bad input format : " + aLines[i]);
                    lErrorMessages.Add("l." + i.ToString() + " Must be : SIMVAR,UNIT");
                }
            }
            init();
        }

        private void SaveFile(bool _bWriteValues, bool bValues)
        {
            Microsoft.Win32.SaveFileDialog oSaveFileDialog = new Microsoft.Win32.SaveFileDialog();
            oSaveFileDialog.Filter = "Simvars files (*.simvars)|*.simvars";
            if (oSaveFileDialog.ShowDialog() == true)
            {
                using (StreamWriter oStreamWriter = new StreamWriter(oSaveFileDialog.FileName, false))
                {
                    foreach (SimvarRequest oSimvarRequest in lSimvarRequests)
                    {
                        // Format : Simvar,Unit
                        string sFormatedLine = oSimvarRequest.sName + "," + oSimvarRequest.sUnits + "," + oSimvarRequest.bIsString;
                        if (bValues)
                        {
                            sFormatedLine += ",  " + oSimvarRequest.dValue.ToString();
                        }
                        oStreamWriter.WriteLine(sFormatedLine);
                    }
                }
            }
        }
    }
}
