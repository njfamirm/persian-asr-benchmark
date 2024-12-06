dictionary_mapping = {
    'ك': 'ک', 'دِ': 'د', 'بِ': 'ب', 'زِ': 'ز', 'ذِ': 'ذ', 'شِ': 'ش', 'سِ': 'س', 'ى': 'ی',
    'ي': 'ی', 'أ': 'ا', 'ؤ': 'و', "ے": "ی", "ۀ": "ه", "ﭘ": "پ", "ﮐ": "ک", "ﯽ": "ی",
    "ﺎ": "ا", "ﺑ": "ب", "ﺘ": "ت", "ﺧ": "خ", "ﺩ": "د", "ﺱ": "س", "ﻀ": "ض", "ﻌ": "ع",
    "ﻟ": "ل", "ﻡ": "م", "ﻢ": "م", "ﻪ": "ه", "ﻮ": "و", 'ﺍ': "ا", 'ة': "ه",
    'ﯾ': "ی", 'ﯿ': "ی", 'ﺒ': "ب", 'ﺖ': "ت", 'ﺪ': "د", 'ﺮ': "ر", 'ﺴ': "س", 'ﺷ': "ش",
    'ﺸ': "ش", 'ﻋ': "ع", 'ﻤ': "م", 'ﻥ': "ن", 'ﻧ': "ن", 'ﻭ': "و", 'ﺭ': "ر", "ﮔ": "گ",

    "a": "‌ای‌", "b": "‌بی‌", "c": "‌سی‌", "d": "‌دی‌", "e": "‌ایی‌", "f": "‌اف‌",
    "g": "‌جی‌", "h": "‌اچ‌", "i": "‌آی‌", "j": "‌جی‌", "k": "‌کی‌", "l": "‌ال‌",
    "m": "‌ام‌", "n": "‌ان‌", "o": "‌او‌", "p": "‌پی‌", "q": "‌کیو‌", "r": "‌آر‌",
    "s": "‌اس‌", "t": "‌تی‌", "u": "‌یو‌", "v": "‌وی‌", "w": "‌دبلیو‌", "x": "‌اکس‌",
    "y": "‌وای‌", "z": "‌زد ",
    "\u200c": " ", "\u200d": " ", "\u200e": " ", "\u200f": " ", "\ufeff": " ",

    "نو آوری‌مان": "نو‌آوری‌مان",
    "نو آوری مان": "نو‌آوری‌مان",
    "نو آوریمان": "نو‌آوری‌مان",
    " ا م ": "‌ام ",
    " م ": "‌ام ",
    "کنندهای": "کننده‌ای",
    "ارائهای": "ارائه‌ای",
    "ایدهای": "ایده‌ای",
    "ماسهای": "ماسه‌ای",
    "خامنهای": "خامنه‌ای",
    "قلهای": "قله‌ای",
    "سیارهای": "سیاره‌ای",
    "کیسهای": "کیسه‌ای",
    "شانهای": "شانه‌ای",
    "غریبهای": "غریبه‌ای",
    "برنامهای": "برنامه‌ای",
    "سختگیرانهای": "سختگیرانه‌ای",
    "بهانهای": "بهانه‌ای",
    "زیرروالهای": "زیر روالهای",
    "درهای": "دره‌ای",
    "آمادهای": "آماده‌ای",
    "سادهای": "ساده‌ای",
    "سرمایهگذارهای": "سرمایه گذارهای",
    "فوقالعادهای": "فوق‌العاده‌ای",
    "حادثهای": "حادثه‌ای",
    "نویسندههای": "نویسنده‌های",
    "علاقهای": "علاقه‌ای",
    "برجستهای": "برجسته‌ای",
    "جلگهای": "جلگه‌ای",
    "زندهای": "زنده‌ای",
    "فنآوریهای": "فناوری‌های",
    "سایهروشنهای": "سایه روشن‌های",
    "بیسابقهای": "بی سابقه‌ای",
    "فرضیهای": "فرضیه‌ای",
    "راهاندازهای": "راه اندازهای",
    "بیشهای": "بیشه‌ای",
    "مقالهای": "مقاله‌ای",
    "دیگهای": "دیگه‌ای",
    "ماههاست": "ماه هاست",
    "نرمافزارهای": "نرم‌افزارهای",
    "کتابسوزانهای": "کتاب سوزان‌های",
    "سیستمعاملهای": "سیستم عامل‌های",
    "اسلحهای": "اسلحه‌ای",
    "وقفهای": "وقفه‌ای",
    "زمینهای": "زمینه‌ای",
    "حرامزادههای": "حرامزاده‌های",
    "هزینهای": "هزینه‌ای",
    "انداختهای": "انداخته‌ای",
    "جسورانهای": "جسورانه‌ای",
    "فاجعهای": "فاجعه‌ای",
    "جامعهای": "جامعه‌ای",
    "پدیدهای": "پدیده‌ای",
    "اغواگرانهای": "اغواگرانه‌ای",
    "تکانهای": "تکانه‌ای",
    "لولهای": "لوله‌ای",
    "نشانهای": "نشانه‌ای",
    "وسیلهای": "وسیله‌ای",
    "آیندهای": "آینده‌ای",
    "بردهای": "برده‌ای",
    "سابقهای": "سابقه‌ای",
    "ناحیهای": "ناحیه‌ای",
    "تکاندهندهای": "تکان دهنده‌ای",
    "بودجهای": "بودجه‌ای",
    "روزانهای": "روزانه‌ای",
    "چارهای": "چاره‌ای",
    "انگیزهای": "انگیزه‌ای",
    "دادهای": "داده‌ای",
    "عدهای": "عده‌ای",
    "هفتهای": "هفته‌ای",
    "منطقهای": "منطقه‌ای",
    "استارتآپهای": "استارتاپ‌های",
    "سازهای": "سازه‌ای",
    "مجموعهای": "مجموعه‌ای",
    "فلسفهای": "فلسفه‌ای",
    "تذکردهندهای": "تذکر دهنده‌ای",
    "مصاحبهای": "مصابحه‌ای",
    "نمونهای": "نمونه‌ای",
    "قلمموهای": "قلم مو‌های",
    "شبزندهداری": "شب زنده‌داری",
    "خوردهباشد": "خورده باشد",
    "داشتهباشید": "داشته باشید",
    "فزایندهای": "فزاینده‌ای",
    "عمدهای": "عمده‌ای",
    "بدیهایی": "بدی‌های",
    "نوشت‌هایم": "نوشته‌ایم",
    "بنتالهدی": "بنت الهدی",
    "نوشتهام": "نوشته‌ام",
    "سرمایهگذاران": "سرمایه گذاران",
    "خانهی": "خانه‌ی",
    "گستاخانهی": "گستاخانه‌ی",
    "گرفتهباشیم": "گرفته باشیم",
    "خونهی": "خونه‌ی",
    "داشتهام": "داشته‌ام",
    "رشتهام": "رشته‌ام",
    "سرمایهگذارانشان": "سرمایه گذارانشان",
    "ریشهکنی": "ریشه‌کنی",
    "مودبانهتری": "مودبانه‌تری",
    "برگردانشدهاند": "برگردان شده‌اند",
    "قرمهسبزی": "قرمه‌سبزی",
    "راهجویی": "راه جویی",
    "اماهیچوقت": "اما هیچوقت",
    "آبوهوای": "آب و هوای",
    "بقیهاش": "بقیه‌اش",
    "طبقهبندی": "طبقه‌بندی",
    "مردههان": "مرده هان",
    "آمادهاند": "آماده‌اند",
    "نشدهاید": "نشده‌اید",
    "آگاهیرسانی": "آگاهی رسانی",
    "نداشتهاند": "نداشته‌اند",
    "شکنانهترین": "شکنانه‌ترین",
    "اقدامهایی": "اقدام‌هایی",
    "راهآهن": "راه آهن",
    "شدهاند": "شده‌اند",
    "تازهترین": "تازه‌ترین",
    "روبهروی": "رو به روی",
    "منحصربهفرد": "منحصر به فرد",
    "سیزدهبدر": "سیزده بدر",
    "برندهی": "برنده‌ی",
    "خانهاشتراکی": "خانه اشتراکی",
    "دادههایی": "داده‌هایی",
    "استفادهتر": "استفاده‌تر",
    "گذرنامهتان": "گذرنامه‌تان",
    "کهنترین": "کهنه‌ترین",
    "فرهنگسرا": "فرهنگ‌سرا",
    "آمادهاید": "آماده‌اید",
    "ویژهی": "ویژه‌ی",
    "غریزهات": "غریزه‌ات",
    "مادرشوهری": "مادر شوهری",
    "نبودهام": "نبوده‌ام",
    "بودهاند": "بوده‌اند",
    "وتنها": "و تنها",
    "بداههکاری": "بداهه‌کاری",
    "سرمایهگذار": "سرمایه گذار",
    "برنامهنویس": "برنامه نویس",
    "مهنازخانم": "مهناز خانم",
    "مواجهاند": "مواجه‌اند",
    "توسعهاش": "توسعه‌اش",
    "سینهام": "سینه‌ام",
    "سین‌هام": "سینه‌ام",
    "نمیخواهند": "نمیخواهند",
    "فنآوری‌ها": "فناوری‌ها",
    "دنبالهرو": "دنباله‌رو",
    "لبهی": "لبه‌ی",
    "اللهیار": "الله یار",
    "ارزندهتر": "ارزنده‌تر",
    "برههای": "بره‌ای",
    "پیادهسازی": "پیاده‌سازی",
    "دهسالگی": "ده سالگی",
    "رسانهای": "رسانه‌ای",
    "ریشسفیدها": "ریش سفید‌ها",
    "چهجوری": "چه جوری",
    "ویژگیهایی": "ویژگی‌هایی",
    "می‌فهمی‌م": "میفهمیم",
    "وبهم": "و بهم",
    "قطرهای": "قطره‌ای",
    "ازتنهایی": "از تنهایی",
    "لطیفهای": "لطیفه‌ای",
    "باشهاومدم": "باشه اومدم",
    "منحصربهفردترین": "منحصر به فرد‌ترین",
    "کردهاند": "کرده‌اند",
    "اندازهای": "اندازه‌ای",
    "بهرهبرداری": "بهره برداری",
    "اماشوهرجان": "اما شوهر جان",
    "خانوادهاش": "خانواده‌اش",
    "نشدهاند": "نشده‌اند",
    "نکردهایم": "نکرده‌ایم",
    "تخممرغ‌هایش": "تخم مرغ‌هایش",
    "وظیفهش": "وظیفه‌اش",
    "مشگینشهر": "مشگی شهر",
    "توسعهدهندگانش": "توسعه دهندگانش",
    "امینابراهیم": "امین ابراهیم",
    "دربارهاش": "درباره‌اش",
    "میانافزارها": "میان‌افزارها",
    "دیدهاند": "دیده‌اند",
    "خانوادهام": "خانواده‌ام",
    "مایهی": "مایه‌ی",
    "نوشتهشدن": "نوشته شدن",
    "راهحل‌هایشان": "راه حل‌هایشان",
    "میهماننواز": "میهمان نواز",
    "زیبندهی": "زیرنده‌ی",
    "راههایی": "راه‌هایی",
    "جربزهی": "جربزه‌ی",
    "بهجا": " به جا",
    "بطورهمزمان": "به طور همزمان",
    "فهمیدهبود": "فهمیده بود",
    "دوربرگردان‌ها": "دور برگردان‌ها",
    "شالودهی": "شالوده‌ی",
    "راهکاریی": "راه‌کاری",
    "مخالفتهایی": "مخالفت‌هایی",
    "چیزهاازشون": "چیزها ازشون",
    "سکونتگاه‌های": "سکونت گاه‌های",
    "سالهابود": "سال‌ها بود",
    "نمونهی": "نمونه‌ی",
    "سرمایهگذاری": "سرمایه گذاری",
    "شبکهای": "شبکه‌ای",
    "خواهرشوهر": "خواهر شوهر",
    "سرگیجهآور": "سرگیجه آور",
    "آستانهی": "آستانه‌ی",
    "دادهاست": "داده است",
    "مجسمهسازی": "مجسمه سازی",
    "ماهرانهترین": "ماهرانه‌ترین",
    "پنجشنبههایی": "پنجشنبه شب‌هایی",
    "نرفنهام": "نرفته‌ام",
    "قورمهسبزی": "قورمه سبزی",
    "گذارهای": "گذاره‌ای",
    "بندهخدا": "بنده خدا",
    "روزنامهنگاران": "روزنامه نگاران",
    "نقشهی": "نقشه‌ی",
    "حملهی": "حمله‌ی",
    "تکنیکهاست": "تکنیک هاست",
    "نرمافزارهایمان": "نرم‌افرارهایمان",
    "مادرشوهرم": "مادر شوهرم",
    "ماهگیمون": "ماه گیمون",
    "مادرشوهرمحترم": "مادر شوهر محترم",
    "شوهرداری": "شوهر داری",
    "سرمایهگذارها": "سرمایه گذارها",
    "بهرهمند": "بهره‌مند",
    "درمانهایی": "درمان‌هایی",
    "عامدانهتر": "عامدانه‌تر",
    "تازهوارد": "تازه وارد",
    "مونتهویدئو": "مونته ویدئو",
    "ذائق‌هاش": "ذائقه‌اش",
    "گوشهگیرتر": "گوشه‌گیرتر",
    "دنبالهدار": "دنباله‌دار",
    "بیخانمان‌ها": "بی‌خانمان‌ها",
    "سرمایهدارها": "سرمایه‌دارها",
    "مادرشوهریم": "مادر شوهریم",
    "صبحان‌هاش": "صبحانه‌اش",
    "جنازهست": "جنازه است",
    "شمارهات": "شماره‌ای",
    "بهقدری": "به قدری",
    "کیسهی": "کیسه‌ی",
    "کوششهایی": "کوشش‌هایی",
    "مادرشوهر": "مادر شوهر",
    "رابطهی": "رابطه‌ی",
    "نوشتهاند": "نوشته‌اند",
    "کنجکاوانهی": "کنجکاوانه‌ی",
    "غیرمتعهد": "غیر متعهد",
    "کردهای": "کرده‌ای",
    "وهمکارانم": "و همکارانم",
    "گردهمآیی": "گردهمایی",
    "اللهوردی": "الله وردی",
    "صرفهجویی": "صرفه جویی",
    "ماندهاند": "مانده‌اند",
    "برنامهنویسی": "برنامه‌نویسی",
    "امینمهدی": "امین مهدی",
    "سهامدارنی": "سهام دارانی",
    "مسابقهی": "مسابقه‌ی",
    "ستارهشناسم": "ستار شناسم",
    "گرفتهاند": "گرفته‌اند",
    "جامعهشان": "جامعه‌شان",
    "بچهی": "بچه‌ی",
    "شیوهی": "شیوه‌ی",
    "بهکار": "به کار",
    "بهتراست": "بهتر است",
    "سروکلهشون": "سر و کلهشون",
    "رسیدهمسرش": "رسید همسرش",
    "پسراهل": "پسر اهل",
    "پروژههای": "پروژه‌های",
    "عاقلان‌هام": "عاقلانه‌ام",
    "گذاشتهاند": "گذاشته‌اند",
    "کردهام": "کرده‌ام",
    "اندازهگیری": "اندازه گیری",
    "یاوهگویی": "یاوه گویی",
    "سازمانهایی": "سازمان‌هایی",
    "نمودهاند": "نموده‌اند",
    "تنهاییآور": "تنهایی آور",
    "قراردهیم": "قرار دهیم",
    "ازشوهرجان": "از شوهر جان",
    "کرهجنوبی": "کره جنوبی",
    "توهینآمیز": "توهین آمیز",
    "فنآوریهایی": "فناوری‌هایی",
    "داشتهاید": "داشته‌اید",
    "شدهایم": "شده‌ایم",
    "نمیفهمم": "نمیفهمم",
    "مثالهایی": "مثال‌هایی",
    "رییسجمهور": "رییس جمهور",
    "مجموعهی": "مجموعه‌ی",
    "درندهاند": "درنده‌اند",
    "امابهش": "اما بهش",
    "بازخواهند": "باز خواهند",
    "برنامههایی": "برنامه‌هایی",
    "یهجا": "یه جا",
    "زگیلهایی": "زگیل‌هایی",
    "وسیلهی": "وسیله‌ی",
    "بهمنیار": "بهمن یار",
    "دادهام": "داده‌ام",
    "بههنگام": "به هنگام",
    "بهدروغ": "به دروغ",
    "دورافتادهترین": "دور افتاده‌ترین",
    "نامهایی": "نامه‌ایی",
    "سهقسمتی": "سه قسمتی",
    "توجهازچیدن": "توجه از چیدن",
    "پیامرسان‌ها": "پیام رسان‌ها",
    "بهمنزاد": "بهمن زاد",
    "نشانههایی": "نشانه‌هایی",
    "راهحل‌های": "راه حل‌های",
    "راهحلهایی": "راه حل‌هایی",
    "راهحلهای": "راه حل‌های",
    "نظرخواهی‌ها": "نظر خواهی‌ها",
    "نظرخواهیها": "نظر خواهی‌ها",
    "کندهی": "کنده‌ی",
    "حرامزاده‌های": "حرام زاده‌های",
    "شبیهسازیهایی": "شبیه سازی‌هایی",
    "مهارتهایی": "مهارت‌هایی",
    "روبهرویشان": "رو به رویشان",
    "برجستهترین": "برجسته‌ترین",
    "نمیفهمیدم": "نمیفهمیدم",
    "دستگاههایی": "دستگاه‌هایی",
    "برادرشوهر": "برادر شوهر",
    "گرسن‌هام": "گرسته‌ام",
    "گرسنههام": "گرسته‌ام",
    "قهوهخوری": "قهوه خوری",
    "دادهاید": "داده‌اید",
    "بهآرامی": "به آرمانی",
    "دانستنیهاست": "دانستنی‌هاست",
    "بهراحتی": "به راحتی",
    "ایدهپردازی": "ایده‌پردازی",
    "ریشسفیدهای": "ریش سفید‌های",
    "خفهمون": "خفه مون",
    "بهجای": "به جای",
    "ریزخشونت‌ها": "ریز خشونت‌ها",
    "ریزخشونتها": "ریز خشونت‌ها",
    "حساسیتهایی": "حساسیت‌هایی",
    "پشتصحنهی": "پشت صحنه‌ی",
    "کلهی": "کله‌ی",
    "تاشوهرم": "تا شوهرم",
    "آیندهاش": "آینده‌اش",
    "پروانههایی": "پروانه‌هایی",
    "خوبیهایی": "خوبی‌هایی",
    "نرمافزارها": "نرم‌افزارها",
    "رساندهاند": "رسانده‌اند",
    "سرمایهگذارنی": "سرمایه گذارانی",
    "تکهچسبانی": "تکه چسبانی",
    "بیتوجهی": "بی توجهی",
    "جاهطلبی": "جاه طلبی",
    "پرغلغلهتان": "پر غلغله‌تان",
    "خمینیشهر": "خمینی شهر",
    "رشتهتوییت": "رشته توییت",
    "موهبتهایی": "موهبت‌هایی",
    "برنامهی": "برنامه‌ی",
    "مادرشوهردارم": "مادر شوهر داردم",
    "سیاهپوستان": "سیاه پوستان",
    "شرکتهایی": "شرکت‌هایی",
    "نیاوردهاند": "نیاورده‌اند",
    "آنهم": "آن هم",
    "شوهرداریم": "شوهر داریم",
    "یکچهارم": "یک چهارم",
    "پروندههاست": "پرونده هاست",
    "برنامهت": "برنامه‌ات",
    "چروکیدهمان": "چروکیده‌مان",
    "زمینهسازی": "زمینه سازی",
    "زدهاند": "زده‌اند",
    "اظهارنظرپرداختن": "اظهار نظر پرداختن",
    "صلحطلبانهترین": "صلح طلبانه‌ترین",
    "بهغلط": "به غلط",
    "ایدهآلم": "ایده آلم",
    "سیاهکاران": "سیاه کاران",
    "امیرابراهیم": "امیر ابراهیم",
    "توسعهدهندگان": "توسعه دهندگان",
    "لحظهی": "لحظه‌ی",
    "امینطاها": "امین طاها",
    "بینالنهرین": "بین النهرین",
    "نیمهوقت": "نیمه وقت",
    "پیادهروی": "پیاده روی",
    "آلودهاند": "آلوده‌اند",
    "گریهکرد": "گره کرد",
    "نعمتهایی": "نعمت‌هایی",
    "مادرشوهرشماهم": "مادر شوهر شما هم",
    "آشپزخونهاس": "آشپزخونه‌اس",
    "مسابقهها": "مسابقه‌ها",
    "مسابقهای": "مسابقه‌های",
    "برنامهریزی": "برنامه‌ریزی",
    "بازخواهید": "باز خواهید",
    "جوییما": "جویی ما",
    "آماده ایم": "آماده‌ایم",
    "مدلسازی": "مدل‌سازی",
    "درصورتیکه": "در صورتیکه",
    "آمریکاییات": "آمریکایی‌ات",
    "مادریاش": "مادری‌اش",
    "غافلگیرکننده": "غافلگیر کننده",
    "پیکرتراشی": "پیکر تراشی",
    "اذیتوآزار": "اذیت و آزار",
    "امتیازاورترین": "امتیاز آور",
    "جیکجیک": "جیک جیک",
    "تاشب": "تا شب",
    "کپیرایت": "کپی رایت",
    "آنتیبادی": "آنتی بادی",
    "عجیبتر": "عجیب‌تر",
    "استانداردسازی": "استاندارد سازی",
    "هشتادوهشت": "هشتاد و هشت",
    "متنوعتر": "متنوع‌تر",
    "منظورانجام": "منظور انجام",
    "نگرانکننده‌ترین": "نگران کننده‌ترین",
    "شگفتانگیز": "شگفت انگیز",
    "رنگینپوست": "رنگین پوست",
    "فارغ التحصیلان": "فارغ‌التحصیلان",
    "ترسناکتر": "ترسناک‌تر",
    "لا رامبلا": "لارامبلا",
    "پرجمعیتترین": "پرجمعیت‌ترین",
    "درمیآیند": "درمی‌آیند",
    "باشمالکی": "باشم الکی",
    "وسیعتر": "وسیع‌تر",
    "فاحشهخانه": "فاحشه خانه",
    "بااحتیاط": "با احتیاط",
    "قانعکننده": "قانع‌کننده",
    "انعطافپذیری": "انعطاف‌پذیری",
    "بیتالمقدس": "بیت‌المقدس",
    "اوپناستریتمپ": "اوپن استریت مپ",
    "روزابارونی": "روزا بارونی",
    "محافظهکارانه": "محافظه کارانه",
    "فوتبالدستی": "فوتبال دستی",
    "توسعهدهنده": "توسعه دهنده",
    "قانونگزاران": "قانون گزاران",
    "العسریسرا": "العسر یسرا",
    "خارقالعاده": "خارق‌العاده",
    "بیماریمزمن": "بیماری مزمن",
    "بادوستانتان": "با دوستانتان",
    "برابربیشتر": "برابر بیشتر",
    "ارائهدهنده": "ارائه دهنده",
    "طوفانزدگان": "طوفان زندگان",
    "امینمحمد": "امین محمد",
    "محیطزیست": "محیط زیست",
    "شقیترینشان": "شقی‌ترینشان",
    "بودواقعا": "بود واقعا",
    "نیویورکتایمز": "نیویورک تایمز",
    "ریودوژانیرو": "ریو دو ژانیرو",
    "مشترکالمنافع": "مشترک‌المنافع",
    "اسلایدسازم": "اسلاید سازم",
    "نمیآوریدش": "نمی‌آوریدش",
    "بینالملل": "بین‌الملل",
    "مصرفکنندگان": "مصرف کنندگان",
    "امینالدین": "امین الدین",
    "امریکااینقدر": "امریکا اینقدر",
    "بعضیاوقات": "بعضی اوقات",
    "خاطربچه": "خاطر بچه",
    "ایناکیلویی": "اینا کیلویی",
    "ویکیپدیا": "ویکی‌پدیا",
    "مافکرمیکنیم": "ما فکر میکنیم",
    "انگلیسیزبان": "انگلیسی زبان",
    "کلهشون": "کله‌شون",
    "آدمبزرگی": "آرم بزرگی",
    "مر آت مر آه": "مر‌آت مر‌آت",
    "آسیبزد": "آسیب زد",
    "آیآرسی": "آی آرسی",
    "آسیااقیانوسیه": "آسیا اقیانوسیه",
    "آیای": "آیا",
    "میانجنسی": "میان جنسی",
    "میاننسلی": "میان نسلی",
    "میان‌افزار‌ها": "میان افزارها",
    "آییننامه": "آیین‌نامه",
    "ارائهشده": "ارائه‌شده",
    "اشپزخونه": "آشپزخونه",
    "اماعلتشونمیپرسه": "اما علتشو نمیپرسه",
    "امیدوارکننده": "امیدوار کننده",
    "ایالاتمتحده": "ایالات متحده",
    "بااینکه": "با اینکه",
    "بلندپروازانه": "بلند پروازانه",
    "بهترازاینه": "بهتر از اینه",
    "بهدست‌آمده": "به دست‌آمده",
    "بهوسیله": "به وسیله",
    "بیادبانه": "بی ادبانه",
    "بیاندازه": "بی اندازه",
    "بیصبرانه": "بی صبرانه",
    "بیفایده": "بی فایده",
    "بیمهره": "بی مهره",
    "بینظیره": "بی نظیره",
    "تاریخزده": "تاریخ زده",
    "تهرانزده": "تهران زده",
    "تولیدشده": "تولید شده",
    "تولیدکننده": "تولید کننده",
    "تکمیلشده": "تکمیل شده",
    "جاافتاده": "جا افتاده",
    "جمع‌آوریکننده": "جمع‌ آوری کننده",
    "جورآدمیه": "جور آدمیه",
    "حقالزحمه": "حق الزحمه",
    "دخترونهتره": "دخترونه تره",
    "دوپنجره": "دو پنجره",
    "ذاتالریه": "ذات‌الریه",
    "راسالخیمه": "راس‌الخیمه",
    "رنگماده": "رنگ ماده",
    "سوئاستفاده": "سو استفاده",
    "سواستفاده": "سو استفاده",
    "شبهجزیره": "شبه جزیره",
    "صادرکننده": "صادر کننده",
    "ضررداره": "ضرر داره",
    "عابرپیاده": "عابر پیاده",
    "فوقالعاده": "فوق‌العاده",
    "قابلتوجه": "قابل توجه",
    "قانع‌کننده": "قانع‌ کننده",
    "مادربیچاره": "مادر بیچاره",
    "مشخصشده": "مشخص شده",
    "مصرفکننده": "مصرف کننده",
    "مصیبتزده": "مصیب تزده",
    "ناامیدکننده": "ناامید کننده",
    "نیمفاصله": "نیم‌فاصله",
    "هماهنگکننده": "هماهنگ کننده",
    "همهجانبه": "همه جانبه",
    "واردکننده": "وارد کننده",
    "وخوابگاه": "و خوابگاه",
    "ودستگاه": "و دستگاه",
    "وزردچوبه": "و زردچوبه",
    "وپروانه": "و پروانه",
    "پدرخوانده": "پدر خوانده",
    "چاپشده": "چاپ شده",
    "کردته": "کرد ته",
    "کردندکه": "کردند که",
    "یکطرفه": "یک طرفه",
    "پایینتره": "پایین‌تره",
    "اشتراکگذاری": "اشتراک گذاری",
    "انحصارگراناند": "انحصار گران‌اند",
    "خوشحالییییی": "خوشحالی",
    "همتیمی‌هایشان": "هم تیمی‌هایشان",
    "پایدار‌ام‌باید": "پایدار‌ام ‌باید",
    "پرجنبوجوش‌تر": "پر جنب و جوش‌تر",
    "آبمروارید": "آب مروارید",
    "آتشسوزی": "آتش سوزی",
    "آتشنشانی": "آتش‌نشانی",
    "آتشنشان": "آتش‌نشان",
    "آرامشبخش": "آرامش بخش",
    "آشناداشتن": "آشنا داشتن",
    "آقاچیزی": "آقا چیزی",
    "آموخت‌هام": "آموخته‌ام",
    "آموزششان": "آموزش‌شان",
    "ازآنجا": "از آنجا",
    "ازالان": "از الان",
    "ازاینجا": "از اینجا",
    "ازجیبش": "از جیبش",
    "ازدستش": "از دستش",
    "ازدیوار": "از دیوار",
    "ازشغلشون": "از شغلشون",
    "ازوقتی": "از وقتی",
    "ازکسانی": "از کسانی",
    "اسباببازی": "اسباب بازی",
    "اسبسوار": "اسب سوار",
    "اصیلزاده": "اصیل زاده",
    "افتادهاید": "افتاده‌اید",
    "ال‌هام": "الهام",
    "امااصلا": "اما اصلا",
    "امااصلابه": "اما اصلا به",
    "امااین": "اما این",
    "امابعد": "اما بعد",
    "امابعدیکی": "اما بعد یکی",
    "اماجاذبه": "اما جاذبه",
    "امرارمعاش": "امرار معاش",
    "امکانپذیر": "امکان پذیر",
    "انت‌های": "انتهای",
    "انت‌هایی": "انتهایی",
    "ایزدبانوی": "ایزد بانوی",
    "بااینحال": "با اینحال",
    "باحتمال": "به احتمال",
    "باحجاب": "با حجاب",
    "باخنده": "با خنده",
    "بادوستاش": "با دوستاش",
    "بارمان": "بار مان",
    "باز‌تر": "باز ‌تر",
    "باطعنه": "با طعنه",
    "بافریاد": "با فریاد",
    "بارگزاری": "بارگذاری",
    "بالامنم": "بالا منم",
    "بگیرمامان": "بگیر مامان",
    "بیاحترامی": "بی احترامی",
    "بیادبی": "بی ادبی",
    "بیاعتنا": "بی اعتنا",
    "بیدارباش": "بیدار باش",
    "بیشازحد": "بیش از حد",
    "بیمسئولیت": "بی مسئولیت",
    "تاسفبار": "تاسف بار",
    "تامشکلمون": "تا مشکلمون",
    "تانقشه": "تا نقشه",
    "تصمیمگیری": "تصمیم گیری",
    "تقسیمبندی": "تقسیم بندی",
    "تقصیرارو": "تقصیرا رو",
    "جدیدابرای": "جدیدا برای",
    "جعبهابزار": "جعبه ابزار",
    "جلوتونو": "جلو تو نو",
    "حاضردر": "حاضر در",
    "حاضرنیست": "حاضر نیست",
    "دستنخورده": "دست نخورده",
    "دوامتیاز": "دو امتیاز",
    "دوروزتمام": "دو روز تمام",
    "شخصیسازی": "شخصی‌سازی",
    "شدواجناس": "شد و اجناس",
    "شوهردارم": "شوهر دارم",
    "شوهرشماهم": "شوهر شما هم",
    "شوهرمحترم": "شوهر محترم",
    "شکلگیری": "شکل گیری",
    "صخرهنوردی": "صخره‌نوردی",
    "صدوبیست": "صد و بیست",
    "عقبنشینی": "عقب نشینی",
    "عکسالعمل": "عکس‌العمل",
    "غرغرمیکنم": "غرغر میکنم",
    "هزاربار": "هزار بار",
    "هزارتومان": "هزار تومان",
    "هزارجور": "هزار جور",
    "هزاروسیصد": "هزار و سیصد",
    "هممیهنان": "هم میهنان",
    "هممیهنانش": "هم میهنانش",
    "همنسلانش": "هم نسلانش",
    "همهگیری": "همه گیری",
    "هییییچ": "هیچ",
    "وقتاخیلی": "وقتا خیلی",
    "وقتابه": "وقتا به",
    "وقتگذرانی": "وقت گذرانی",
    "ومحکوم": "و محکوم",
    "ومحیط‌ها": "و محیط‌ها",
    "وکشورتان": "و کشورتان",
    "ویکیمدیا": "ویکی‌مدیا",
    "یهوگفت": "یهو گفت",
    "اینجااز": "اینجا از",
}
fixator_dictionary = {
    "ب‌های": "بهای",
    "به‌ترین": "بهترین",
    "آس‌تر": "‌آستر",
    "ارکس‌تر": "ارکستر",
    "ان‌تر": "انتر",
    "بس‌تر": "بستر",
    "به‌تر": "بهتر",
    "به‌ترتر": "بهترتر",
    "توئی‌تر": "تویتتر",
    "تویی‌تر": "توییتر",
    "تی‌تر": "تیتر",
    "دخ‌تر": "دختر",
    "دف‌تر": "دفتر",
    "دلس‌تر": "دلستر",
    "دک‌تر": "دکتر",
    "ش‌تر": "شتر",
    "لی‌تر": "لیتر",
    "م‌تر": "متر",
    "هیپس‌تر": "هیپستر",
    "پی‌تر": "پیتر",
    "چ‌تر": "چتر",
    "کم‌تر": "کمتر",
    "گنگس‌تر": "گنگستر",
    "انگش‌تر": "انگشتر",
    "سن‌تر": "سنتر",
    "تویت‌تر": "توییتر",
    "مادهش‌تر": "ماده شتر",
    "وی‌ترین": "ویترین",
    "کرونوم‌تر": "کرنومتر",
    "که‌تر": "کهتر",
    "فیل‌تر": "فیلتر",
    "ال‌هام": "الهام",
    "آل‌مان": "آلمان",
    "انت‌های": "انتهای",
    "انت‌هایی": "انتهایی",
    "آموخت‌هام": "آموخته‌ام",
}