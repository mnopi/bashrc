from rc import Base
from rc import info

ic(Base().get_clsname, Base.get_mroattr(Base), Base.get_mrohash(), Base.get_mrorepr(), Base.get_mroslots(), Base.get_propnew('mierda'),
   )


class C(Base):
    __slots__ = ('_mierda', )
    mierda = Base.get_propnew('mierda')

ic(Base().get_clsname, Base.get_mroattr(Base), Base.get_mrohash(), Base.get_mrorepr(), Base.get_mroslots(),
   Base.get_propnew('mierda'), C().mierda, C.mierda, C().get_clsname, C.get_mroattr(C), C.get_mrohash(), C.get_mrorepr(), C.get_mroslots(),
   )


def tests_base():


