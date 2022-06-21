# 処理の状態を管理する
class ProcessPool:
    # デフォルトはFalse。True・FalseでスイッチのON・OFFのように状態を切り替える
    # 状態によって判定(処理の状態がロックされているかどうか。Trueならロックされている。Falseならロックされていない。)
    locking = False

    # lockingの状態を確認して返す処理(処理の状態を判定する)
    @classmethod
    def is_lock(cls):
        return cls.locking

    # lockingをTrueにする(処理の状態をロックする)
    @classmethod
    def lock(cls):
        cls.locking = True

    # lockingをFalseにする(ロックを解除する)
    @classmethod
    def unlock(cls):
        cls.locking = False
