"""Optional ROS 2 link to the MacroMicro teleop (needs a sourced ROS environment).

Follows:
  /trajectory_recorder/recording  (std_msgs/String JSON, latched): the GREY-button
      recording. Starts/stops an NDI recording in the same run folder (NEW_RUN)
      on the recorder's time base.
  /ur16e_touch_teleop/rcm_active   (std_msgs/Bool, latched): turning RCM mode on
      latches the NDI-measured tool tip as the RCM point (the teleop latches its
      own tool tip at the same moment); turning it off clears it.
Publishes (camera frame "ndi", mm):
  /ndi_tracking/tip        geometry_msgs/PointStamped
  /ndi_tracking/rcm_error  std_msgs/Float64, shaft line -> RCM point
"""
import json
import threading


def start(tracker, recording_topic, rcm_topic, log=print):
    import rclpy
    from geometry_msgs.msg import PointStamped
    from rclpy.qos import DurabilityPolicy, QoSProfile
    from std_msgs.msg import Bool, Float64, String

    from rclpy.executors import SingleThreadedExecutor
    from rclpy.signals import SignalHandlerOptions

    # No rclpy signal handlers: Ctrl+C stays a KeyboardInterrupt in `run`, which shuts
    # this down in order
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = rclpy.create_node('ndi_tracking')
    latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    state = {'rcm': None}

    def on_recording(msg):
        try:
            d = json.loads(msg.data)
        except ValueError:
            log(f'bad message on {recording_topic}: {msg.data!r}')
            return
        if d.get('recording'):
            if not d.get('run_dir'):
                log('teleop recording started without a run folder (its logs failed): '
                    'recording to ndi_tracking/recordings instead')
            log(tracker.start_recording(d.get('run_dir'), d.get('start_unix'), source='teleop'))
        elif tracker.session and tracker.session.source == 'teleop':
            log(tracker.stop_recording(source='teleop'))

    def on_rcm(msg):
        first, changed = state['rcm'] is None, state['rcm'] != msg.data
        state['rcm'] = msg.data
        if not changed:
            return
        if first:
            if msg.data:
                log('RCM mode was already on when the NDI tracker started: its point is unknown here. '
                    "Press 'r' with the tool tip at the RCM point, or leave and re-enter RCM mode")
            return
        if msg.data:
            log(tracker.request_rcm('teleop'))
        else:
            log(tracker.set_rcm(None, 'teleop'))

    node.create_subscription(String, recording_topic, on_recording, latched)
    node.create_subscription(Bool, rcm_topic, on_rcm, latched)
    tip_pub = node.create_publisher(PointStamped, '~/tip', 10)
    err_pub = node.create_publisher(Float64, '~/rcm_error', 10)

    def publish():
        latest = tracker.latest
        if latest is None or latest[2] is None:
            return
        _, _, tip, _, err, _ = latest
        m = PointStamped()
        m.header.stamp = node.get_clock().now().to_msg()
        m.header.frame_id = 'ndi'
        m.point.x, m.point.y, m.point.z = (float(v) for v in tip)
        tip_pub.publish(m)
        if err == err:      # not NaN
            err_pub.publish(Float64(data=err))

    node.create_timer(1.0 / tracker.rate, publish)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    stop = threading.Event()

    def spin():
        while not stop.is_set():
            try:
                executor.spin_once(timeout_sec=0.1)
            except Exception:   # shutdown in progress
                break
    thread = threading.Thread(target=spin, daemon=True)
    thread.start()
    log(f'ROS link up: following {recording_topic} and {rcm_topic}')

    def shutdown():
        # Stop and join the spin thread *before* tearing ROS down: a thread still inside
        # rclpy at interpreter exit aborts Python ("terminate called without an active
        # exception"), which Ubuntu reports as a crash
        stop.set()
        thread.join(timeout=2.0)
        executor.shutdown(timeout_sec=1.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return shutdown
